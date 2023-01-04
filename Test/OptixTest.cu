#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <array>

#include "OptixTest.hpp"
#include "AssetLoader.hpp"
#include "Image.hpp"

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenSbtRecord   =   SbtRecord<RayGenData>;
using MissSbtRecord     =   SbtRecord<MissData>;
using HitGroupSbtRecord =   SbtRecord<HitGroupData>;

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

__global__ void rand_init(curandStateMRG32k3a *rand_state, const unsigned int max_x, const unsigned int max_y) {
    // Each thread in a block gets unique seed
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    unsigned int pixel_index = j * max_x + i;
    curand_init(2023 + pixel_index, 0, 0, &rand_state[pixel_index]);
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

    // accel handling
    OptixTraversableHandle  gas_handle;
    CUdeviceptr             d_gas_output_buffer;
    {
        OptixAccelBuildOptions  accel_options = {};
        accel_options.buildFlags    =   OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation     =   OPTIX_BUILD_OPERATION_BUILD;

        std::array<float3, 2>  sphereVertex   =   { make_float3(0.0f, 0.0f, -1.0f), make_float3(0, -100.5, -1)};
        size_t sphereVertexSize = sizeof(sphereVertex[0]) * sphereVertex.size();
        std::array<float,  2>  sphereRadius   =   { 0.5f, 100.f };
        size_t sphereRadiusSize = sizeof(sphereRadius[0]) * sphereRadius.size();

        CUdeviceptr d_vertex_buffer;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_vertex_buffer), sphereVertexSize));
        checkCudaErrors(cudaMemcpy(reinterpret_cast<void **>(d_vertex_buffer), sphereVertex.data(), sphereVertexSize, cudaMemcpyHostToDevice));

        CUdeviceptr d_radius_buffer;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_radius_buffer), sphereRadiusSize));
        checkCudaErrors(cudaMemcpy(reinterpret_cast<void *>(d_radius_buffer), sphereRadius.data(), sphereRadiusSize, cudaMemcpyHostToDevice));

        OptixBuildInput sphere_input = {};

        sphere_input.type                       = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        sphere_input.sphereArray.vertexBuffers  = &d_vertex_buffer;
        sphere_input.sphereArray.numVertices    = sphereVertex.size();
        sphere_input.sphereArray.radiusBuffers  = &d_radius_buffer;

        static std::array<uint32_t, 2> g_mat_indices = { 1, 0 };
        CUdeviceptr d_mat_indices;
        const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof(uint32_t);
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_mat_indices), mat_indices_size_in_bytes));
        checkCudaErrors(cudaMemcpy(
            reinterpret_cast<void *>(d_mat_indices),
            g_mat_indices.data(),
            mat_indices_size_in_bytes,
            cudaMemcpyHostToDevice
        ));

        uint32_t sphere_input_flags[MAT_COUNT]          = {
            OPTIX_GEOMETRY_FLAG_NONE,
            OPTIX_GEOMETRY_FLAG_NONE,
            OPTIX_GEOMETRY_FLAG_NONE
        };
        sphere_input.sphereArray.flags          = sphere_input_flags;
        sphere_input.sphereArray.numSbtRecords  = MAT_COUNT;
        sphere_input.sphereArray.sbtIndexOffsetBuffer       = d_mat_indices;
        sphere_input.sphereArray.sbtIndexOffsetSizeInBytes  = sizeof(uint32_t);
        sphere_input.sphereArray.sbtIndexOffsetStrideInBytes= sizeof(uint32_t);

        OptixAccelBufferSizes   gas_buffer_sizes;
        checkOptiXErrors(optixAccelComputeMemoryUsage(context, &accel_options, &sphere_input, 1, &gas_buffer_sizes));
        CUdeviceptr d_temp_buffer_gas;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

        // non-compacted output
        CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t      compactedSizeOffset = My::roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size), compactedSizeOffset + 8));

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result             = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

        checkOptiXErrors(optixAccelBuild(context,
                                        0,
                                        &accel_options,
                                        &sphere_input,
                                        1,
                                        d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                                        d_buffer_temp_output_gas_and_compacted_size, gas_buffer_sizes.outputSizeInBytes,
                                        &gas_handle,
                                        &emitProperty,
                                        1));

        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;

        checkCudaErrors(cudaFree((void *)d_temp_buffer_gas));
        checkCudaErrors(cudaFree((void *)d_mat_indices));
        checkCudaErrors(cudaFree((void *)d_vertex_buffer));
        checkCudaErrors(cudaFree((void *)d_radius_buffer));

        size_t compacted_gas_size;
        checkCudaErrors(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

        if(compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_gas_output_buffer), compacted_gas_size));

            // use handle as input and output
            checkOptiXErrors(optixAccelCompact(context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

            checkCudaErrors(cudaFree((void *)d_buffer_temp_output_gas_and_compacted_size));
        } else {
            d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    }
    
    // Create module
    OptixModule module = nullptr;
    OptixModule sphere_module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};
#ifdef _DEBUG
        module_compile_options.optLevel     = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel   = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
        pipeline_compile_options.usesMotionBlur         = false;
        pipeline_compile_options.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues       = 11;
        pipeline_compile_options.numAttributeValues     = 1;
        pipeline_compile_options.exceptionFlags         = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

        My::AssetLoader assetLoader;
        auto shader = assetLoader.SyncOpenAndReadBinary("Shaders/CUDA/OptixTest.shader.optixir");

        checkOptiXErrorsLog(optixModuleCreateFromPTX(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            (const char*)shader.GetData(),
            shader.GetDataSize(),
            LOG, &LOG_SIZE,
            &module
        ));

        OptixBuiltinISOptions builtin_is_options = {};

        builtin_is_options.usesMotionBlur       = false;
        builtin_is_options.builtinISModuleType  = OPTIX_PRIMITIVE_TYPE_SPHERE;
        checkOptiXErrors(optixBuiltinISModuleGet(context, &module_compile_options, &pipeline_compile_options,
                                                &builtin_is_options, &sphere_module));
    }

    // Create program groups
    OptixProgramGroup raygen_prog_group         = nullptr;
    OptixProgramGroup miss_prog_group           = nullptr;
    OptixProgramGroup hitgroup_prog_group       = nullptr;
    {
        OptixProgramGroupOptions program_group_options  = {};

        // ray gen group
        OptixProgramGroupDesc    raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        checkOptiXErrorsLog(optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1, // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &raygen_prog_group
        ));

        // miss group
        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

        checkOptiXErrorsLog(optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,
            &program_group_options,
            LOG, &LOG_SIZE,
            &miss_prog_group
        )); 

        // hit group
        OptixProgramGroupDesc hitgroup_prog_group_desc          = {};
        hitgroup_prog_group_desc.kind                           = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH              = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH   = "__closesthit__ch";
        hitgroup_prog_group_desc.hitgroup.moduleAH              = nullptr;
        hitgroup_prog_group_desc.hitgroup.moduleIS              = sphere_module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS   = nullptr;

        checkOptiXErrorsLog(optixProgramGroupCreate(
            context,
            &hitgroup_prog_group_desc,
            1,
            &program_group_options,
            LOG, &LOG_SIZE,
            &hitgroup_prog_group
        ));
    }

    // Link pipeline
    OptixPipeline pipeline = nullptr;
    {
        const uint32_t      max_trace_depth = 1;
        OptixProgramGroup   program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

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
                                                    0, // maxCCDepth
                                                    0, // maxDCDepth
                                                    &direct_callable_stack_size_from_traversal,
                                                    &direct_callable_stack_size_from_state,
                                                    &continuation_stack_size));
        checkOptiXErrors(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                                    direct_callable_stack_size_from_state,
                                                    continuation_stack_size,
                                                    1 /* maxTraversableDepth */ ));
    }

    // Set up shader binding table
    OptixShaderBindingTable sbt = {};
    {
        CUdeviceptr     raygen_record;
        const size_t    raygen_record_size = sizeof(RayGenSbtRecord);
        checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        checkOptiXErrors(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        checkCudaErrors(cudaMemcpy(
            reinterpret_cast<void*>(raygen_record),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice
        ));

        CUdeviceptr miss_record;
        size_t      miss_record_size = sizeof(MissSbtRecord);
        checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
        MissSbtRecord ms_sbt;
        ms_sbt.data.bg_color = {0.5f, 0.7f, 1.0f};
        checkOptiXErrors(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        checkCudaErrors(cudaMemcpy(
            reinterpret_cast<void**>(miss_record),
            &ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice
        ));

        CUdeviceptr hitgroup_record;
        size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size * MAT_COUNT));
        HitGroupSbtRecord hg_sbt[MAT_COUNT];

        checkOptiXErrors(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt[0]));
        hg_sbt[0].data.material_type = Material::MAT_DIFFUSE;
        hg_sbt[0].data.base_color    = {0.5f, 0.5f, 0.5f};
        
        checkOptiXErrors(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt[1]));
        hg_sbt[1].data.material_type = Material::MAT_DIELECTRIC;
        hg_sbt[1].data.ir            = 1.5f;
        
        checkOptiXErrors(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt[2]));
        hg_sbt[2].data.material_type = Material::MAT_METAL;
        hg_sbt[2].data.base_color    = {0.7f, 0.6f, 0.5f};
        hg_sbt[2].data.fuzz          = 0.1f;
        
        checkCudaErrors(cudaMemcpy(
            reinterpret_cast<void *>(hitgroup_record),
            &hg_sbt,
            hitgroup_record_size * MAT_COUNT,
            cudaMemcpyHostToDevice
        ));

        sbt.raygenRecord                = raygen_record;
        sbt.missRecordBase              = miss_record;
        sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
        sbt.missRecordCount             = 1;
        sbt.hitgroupRecordBase          = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount         = MAT_COUNT;
    }

    // Render Settings
    My::Image img;
    My::Image* d_img;
    My::RayTracingCamera<float>* d_camera;
    curandStateMRG32k3a* d_rand_state;
    {
        const float aspect_ratio = 16.0 / 9.0;
        const int image_width = 1920;
        const int image_height = static_cast<int>(image_width / aspect_ratio);

        // Canvas
        img.Width = image_width;
        img.Height = image_height;
        img.bitcount = 96; 
        img.bitdepth = 32;
        img.pixel_format = My::PIXEL_FORMAT::RGB32;
        img.pitch = (img.bitcount >> 3) * img.Width;
        img.compressed = false;
        img.compress_format = My::COMPRESSED_FORMAT::NONE;
        img.data_size = img.Width * img.Height * (img.bitcount >> 3);
        auto num_pixels = image_width * image_height;

        checkCudaErrors(cudaMallocManaged((void **)&img.data, img.data_size));

        checkCudaErrors(cudaMalloc((void **)&d_img, sizeof(My::Image)));
        checkCudaErrors(cudaMemcpy((void *)d_img, &img, sizeof(My::Image), cudaMemcpyHostToDevice));

        My::Point<float> lookfrom{0, 0, 0};
        My::Point<float> lookat{0, 0, -1};
        My::Vector3f vup{0, 1, 0};
        auto dist_to_focus = 1.0f;
        auto aperture = 0.0f;

        My::RayTracingCamera<float> camera (lookfrom, lookat, vup, 90.0f, aspect_ratio,
                                aperture, dist_to_focus);

        checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(My::RayTracingCamera<float>)));
        checkCudaErrors(cudaMemcpy((void **)d_camera, &camera, sizeof(My::RayTracingCamera<float>), cudaMemcpyHostToDevice)); 

        int tile_width = 8;
        int tile_height = 8;

        dim3 blocks((image_width + tile_width - 1) / tile_width, (image_height + tile_height - 1) / tile_height);
        dim3 threads(tile_width, tile_height);

        checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandStateMRG32k3a)));

        rand_init<<<blocks, threads>>>(d_rand_state, image_width, image_height);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
    // launch
    {
        CUstream stream;
        checkCudaErrors(cudaStreamCreate(&stream));

        Params params;
        params.handle       = gas_handle;
        params.image        = d_img;
        params.cam          = d_camera;
        params.rand_state   = d_rand_state;
        params.max_depth    = 50;
        params.num_of_samples = 512;

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
        checkOptiXErrors(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, img.Width, img.Height, 1));
        cudaEventRecord(stop);

        checkCudaErrors(cudaDeviceSynchronize());

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Rendering time: %f ms\n", milliseconds);

        img.SaveTGA("raytracing_optix.tga");
        img.data = nullptr;  // to avoid double free

        // clean up
        {
            checkCudaErrors(cudaFree(reinterpret_cast<void*>(d_rand_state)));
            checkCudaErrors(cudaFree(reinterpret_cast<void*>(d_img)));
            checkCudaErrors(cudaFree(reinterpret_cast<void*>(d_camera)));
            checkCudaErrors(cudaFree(reinterpret_cast<void*>(d_param)));

            checkCudaErrors(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
            checkCudaErrors(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
            checkCudaErrors(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
            checkCudaErrors(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));

            checkOptiXErrors(optixPipelineDestroy(pipeline));
            checkOptiXErrors(optixProgramGroupDestroy(hitgroup_prog_group));
            checkOptiXErrors(optixProgramGroupDestroy(miss_prog_group));
            checkOptiXErrors(optixProgramGroupDestroy(raygen_prog_group));
            checkOptiXErrors(optixModuleDestroy(module));
            checkOptiXErrors(optixModuleDestroy(sphere_module));

            checkOptiXErrors(optixDeviceContextDestroy(context));
        }
    }

    return 0;
}