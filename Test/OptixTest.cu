#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

#include "OptixTest.h"
#include "AssetLoader.hpp"
#include "Image.hpp"

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<int>        MissSbtRecord;

// help functions 
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " (" << cudaGetErrorString(result) << ") "
                  << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#define checkOptiXErrors(val) check_optix((val), #val, __FILE__, __LINE__)
inline void check_optix( OptixResult res, const char* call, const char* file, unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        std::cerr << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
        exit(98);
    }
}

#define checkOptiXErrorsLog(val)                                                        \
    do {                                                                                \
        char    LOG[2048];                                                              \
        size_t  LOG_SIZE = sizeof(LOG);                                                 \
        check_optix_log((val), LOG, sizeof(LOG), LOG_SIZE, #val, __FILE__, __LINE__);   \
    } while (false)
inline void check_optix_log( OptixResult  res,
                           const char*  log,
                           size_t       sizeof_log,
                           size_t       sizeof_log_returned,
                           const char*  call,
                           const char*  file,
                           unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        std::cerr << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
           << log << ( sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "" ) << '\n';
    }
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}

int main() {
    // Initialize CUDA and create OptiX context
    OptixDeviceContext context = nullptr;
    {
        checkCudaErrors(cudaFree(0));

        CUcontext cuCtx = 0;

        checkOptiXErrors(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &context_log_cb;
        options.logCallbackLevel          = 4;
        checkOptiXErrors(optixDeviceContextCreate(cuCtx, &options, &context));
    }
    
    // Create module
    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};
#ifdef _DEBUG
        module_compile_options.optLevel     = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel   = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
        pipeline_compile_options.usesMotionBlur         = false;
        pipeline_compile_options.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues       = 2;
        pipeline_compile_options.numAttributeValues     = 2;
        pipeline_compile_options.exceptionFlags         = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        My::AssetLoader assetLoader;
        auto shader = assetLoader.SyncOpenAndReadBinary("Shaders/CUDA/draw_solid_color.optixir");

        checkOptiXErrorsLog(optixModuleCreateFromPTX(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            (const char*)shader.GetData(),
            shader.GetDataSize(),
            LOG, &LOG_SIZE,
            &module
        ));
    }

    // Create program groups, including NULL miss and hitgroups
    OptixProgramGroup raygen_prog_group     = nullptr;
    OptixProgramGroup miss_prog_group       = nullptr;
    {
        OptixProgramGroupOptions program_group_options  = {};

        OptixProgramGroupDesc    raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__draw_solid_color";
        checkOptiXErrorsLog(optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1, // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &raygen_prog_group
        ));

        // Leave miss group's module and entryfunc name null
        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        checkOptiXErrorsLog(optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,
            &program_group_options,
            LOG, &LOG_SIZE,
            &miss_prog_group
        )); 
    }

    // Link pipeline
    OptixPipeline pipeline = nullptr;
    {
        const uint32_t      max_trace_depth = 0;
        OptixProgramGroup   program_groups[] = { raygen_prog_group };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth     = max_trace_depth;
        pipeline_link_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        checkOptiXErrorsLog(optixPipelineCreate(
            context,
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups) / sizeof(program_groups[0]),
            LOG, &LOG_SIZE,
            &pipeline
        ) );

        OptixStackSizes stack_sizes = {};
        for (auto& prog_group : program_groups) {
            checkOptiXErrors(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        checkOptiXErrors(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                                    0,
                                                    0,
                                                    &direct_callable_stack_size_from_traversal,
                                                    &direct_callable_stack_size_from_state,
                                                    &continuation_stack_size));
        checkOptiXErrors(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                                    direct_callable_stack_size_from_state,
                                                    continuation_stack_size,
                                                    2));
    }

    // Set up shader binding table
    OptixShaderBindingTable sbt = {};
    {
        CUdeviceptr     raygen_record;
        const size_t    raygen_record_size = sizeof(RayGenSbtRecord);
        checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        checkOptiXErrors(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        rg_sbt.data = {0.462f, 0.725f, 0.f};
        checkCudaErrors(cudaMemcpy(
            reinterpret_cast<void*>(raygen_record),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice
        ));

        CUdeviceptr miss_record;
        size_t      miss_record_size = sizeof(MissSbtRecord);
        checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
        RayGenSbtRecord ms_sbt;
        checkOptiXErrors(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        checkCudaErrors(cudaMemcpy(
            reinterpret_cast<void**>(miss_record),
            &ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice
        ));

        sbt.raygenRecord            = raygen_record;
        sbt.missRecordBase          = miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount         = 1;
    }

    // Render Settings
    const float aspect_ratio = 16.0 / 9.0;
    const int image_width = 1920;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // Canvas
    My::Image img;
    img.Width = image_width;
    img.Height = image_height;
    img.bitcount = 32;
    img.bitdepth = 8;
    img.pixel_format = My::PIXEL_FORMAT::RGBA8;
    img.pitch = (img.bitcount >> 3) * img.Width;
    img.compressed = false;
    img.compress_format = My::COMPRESSED_FORMAT::NONE;
    img.data_size = img.Width * img.Height * (img.bitcount >> 3);

    checkCudaErrors(cudaMallocManaged((void **)&img.data, img.data_size));

    // launch
    {
        CUstream stream;
        checkCudaErrors(cudaStreamCreate(&stream));

        Params params;
        params.image        = reinterpret_cast<uchar4*>(img.data);
        params.image_width  = image_width;

        CUdeviceptr d_param;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
        checkCudaErrors(cudaMemcpy(
            reinterpret_cast<void**>(d_param),
            &params, sizeof(params),
            cudaMemcpyHostToDevice
        ));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        checkOptiXErrors(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, image_width, image_height, 1));
        cudaEventRecord(stop);

        checkCudaErrors(cudaDeviceSynchronize());

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Rendering time: %f ms\n", milliseconds);

        img.SaveTGA("raytracing_optix.tga");
        img.data = nullptr;  // to avoid double free

        // clean up
        {
            checkCudaErrors(cudaFree(reinterpret_cast<void*>(d_param)));
            checkCudaErrors(cudaFree(reinterpret_cast<void**>(sbt.raygenRecord)));
            checkCudaErrors(cudaFree(reinterpret_cast<void**>(sbt.missRecordBase)));

            checkOptiXErrors(optixPipelineDestroy(pipeline));
            checkOptiXErrors(optixProgramGroupDestroy(miss_prog_group));
            checkOptiXErrors(optixProgramGroupDestroy(raygen_prog_group));
            checkOptiXErrors(optixModuleDestroy(module));

            checkOptiXErrors(optixDeviceContextDestroy(context));
        }
    }

    return 0;
}