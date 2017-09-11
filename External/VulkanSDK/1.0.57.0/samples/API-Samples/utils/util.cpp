/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 * Copyright (C) 2015-2016 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
VULKAN_SAMPLE_DESCRIPTION
samples utility functions
*/

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include "util.hpp"

#ifdef __ANDROID__
// Android specific include files.
#include <unordered_map>

// Header files.
#include <android_native_app_glue.h>
#include "shaderc/shaderc.hpp"
// Static variable that keeps ANativeWindow and asset manager instances.
static android_app *Android_application = nullptr;
#elif (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
#	include <MoltenGLSLToSPIRVConverter/GLSLToSPIRVConverter.h>
#else
#	include "SPIRV/GlslangToSpv.h"
#endif

// For timestamp code (get_milliseconds)
#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

using namespace std;

#if !(defined(__ANDROID__) || defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
// Android, iOS, and macOS: main() implemented externally to allow access to Objective-C components
int main(int argc, char **argv) { return sample_main(argc, argv); }
#endif

void extract_version(uint32_t version, uint32_t &major, uint32_t &minor, uint32_t &patch) {
    major = version >> 22;
    minor = (version >> 12) & 0x3ff;
    patch = version & 0xfff;
}

string get_file_name(const string &s) {
    char sep = '/';

#ifdef _WIN32
    sep = '\\';
#endif

    // cout << "in get_file_name\n";
    size_t i = s.rfind(sep, s.length());
    if (i != string::npos) {
        return (s.substr(i + 1, s.length() - i));
    }

    return ("");
}

#if !(defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
// iOS & macOS: get_base_data_dir() implemented externally to allow access to Objective-C components
std::string get_base_data_dir() {
#ifdef __ANDROID__
    return "";
#else
    return std::string(VULKAN_SAMPLES_BASE_DIR) + "/API-Samples/data/";
#endif
}
#endif

std::string get_data_dir(std::string filename) {
    std::string basedir = get_base_data_dir();
    // get the base filename
    std::string fname = get_file_name(filename);

    // get the prefix of the base filename, i.e. the part before the dash
    stringstream stream(fname);
    std::string prefix;
    getline(stream, prefix, '-');
    std::string ddir = basedir + prefix;
    return ddir;
}

bool memory_type_from_properties(struct sample_info &info, uint32_t typeBits, VkFlags requirements_mask, uint32_t *typeIndex) {
    // Search memtypes to find first index with those properties
    for (uint32_t i = 0; i < info.memory_properties.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            // Type is available, does it match user properties?
            if ((info.memory_properties.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
                *typeIndex = i;
                return true;
            }
        }
        typeBits >>= 1;
    }
    // No memory types matched, return failure
    return false;
}

void set_image_layout(struct sample_info &info, VkImage image, VkImageAspectFlags aspectMask, VkImageLayout old_image_layout,
                      VkImageLayout new_image_layout, VkPipelineStageFlags src_stages, VkPipelineStageFlags dest_stages) {
    /* DEPENDS on info.cmd and info.queue initialized */

    assert(info.cmd != VK_NULL_HANDLE);
    assert(info.graphics_queue != VK_NULL_HANDLE);

    VkImageMemoryBarrier image_memory_barrier = {};
    image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_memory_barrier.pNext = NULL;
    image_memory_barrier.srcAccessMask = 0;
    image_memory_barrier.dstAccessMask = 0;
    image_memory_barrier.oldLayout = old_image_layout;
    image_memory_barrier.newLayout = new_image_layout;
    image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.image = image;
    image_memory_barrier.subresourceRange.aspectMask = aspectMask;
    image_memory_barrier.subresourceRange.baseMipLevel = 0;
    image_memory_barrier.subresourceRange.levelCount = 1;
    image_memory_barrier.subresourceRange.baseArrayLayer = 0;
    image_memory_barrier.subresourceRange.layerCount = 1;

    switch (old_image_layout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            image_memory_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_PREINITIALIZED:
            image_memory_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            break;

        default:
            break;
    }

    switch (new_image_layout) {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            image_memory_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            image_memory_barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;

        default:
            break;
    }

    vkCmdPipelineBarrier(info.cmd, src_stages, dest_stages, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);
}

bool read_ppm(char const *const filename, int &width, int &height, uint64_t rowPitch, unsigned char *dataPtr) {
    // PPM format expected from http://netpbm.sourceforge.net/doc/ppm.html
    //  1. magic number
    //  2. whitespace
    //  3. width
    //  4. whitespace
    //  5. height
    //  6. whitespace
    //  7. max color value
    //  8. whitespace
    //  7. data

    // Comments are not supported, but are detected and we kick out
    // Only 8 bits per channel is supported
    // If dataPtr is nullptr, only width and height are returned

    // Read in values from the PPM file as characters to check for comments
    char magicStr[3] = {}, heightStr[6] = {}, widthStr[6] = {}, formatStr[6] = {};

#ifndef __ANDROID__
    FILE *fPtr = fopen(filename, "rb");
#else
    FILE *fPtr = AndroidFopen(filename, "rb");
#endif
    if (!fPtr) {
        printf("Bad filename in read_ppm: %s\n", filename);
        return false;
    }

    // Read the four values from file, accounting with any and all whitepace
    fscanf(fPtr, "%s %s %s %s ", magicStr, widthStr, heightStr, formatStr);

    // Kick out if comments present
    if (magicStr[0] == '#' || widthStr[0] == '#' || heightStr[0] == '#' || formatStr[0] == '#') {
        printf("Unhandled comment in PPM file\n");
        return false;
    }

    // Only one magic value is valid
    if (strncmp(magicStr, "P6", sizeof(magicStr))) {
        printf("Unhandled PPM magic number: %s\n", magicStr);
        return false;
    }

    width = atoi(widthStr);
    height = atoi(heightStr);

    // Ensure we got something sane for width/height
    static const int saneDimension = 32768;  //??
    if (width <= 0 || width > saneDimension) {
        printf("Width seems wrong.  Update read_ppm if not: %u\n", width);
        return false;
    }
    if (height <= 0 || height > saneDimension) {
        printf("Height seems wrong.  Update read_ppm if not: %u\n", height);
        return false;
    }

    if (dataPtr == nullptr) {
        // If no destination pointer, caller only wanted dimensions
        return true;
    }

    // Now read the data
    for (int y = 0; y < height; y++) {
        unsigned char *rowPtr = dataPtr;
        for (int x = 0; x < width; x++) {
            fread(rowPtr, 3, 1, fPtr);
            rowPtr[3] = 255; /* Alpha of 1 */
            rowPtr += 4;
        }
        dataPtr += rowPitch;
    }
    fclose(fPtr);

    return true;
}

#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))

void init_glslang() {}

void finalize_glslang() {}

bool GLSLtoSPV(const VkShaderStageFlagBits shader_type, const char *pshader, std::vector<unsigned int> &spirv) {

 	MLNShaderStage shaderStage;
 	switch (shader_type) {
		 		case VK_SHADER_STAGE_VERTEX_BIT:
		 			shaderStage = kMLNShaderStageVertex;
		 			break;
		 		case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
		 			shaderStage = kMLNShaderStageTessControl;
		 			break;
				case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
		 			shaderStage = kMLNShaderStageTessEval;
		 			break;
		 		case VK_SHADER_STAGE_GEOMETRY_BIT:
		 			shaderStage = kMLNShaderStageGeometry;
		 			break;
		 		case VK_SHADER_STAGE_FRAGMENT_BIT:
		 			shaderStage = kMLNShaderStageFragment;
		 			break;
		 		case VK_SHADER_STAGE_COMPUTE_BIT:
		 			shaderStage = kMLNShaderStageCompute;
		 			break;
		 		default:
		 			shaderStage = kMLNShaderStageAuto;
		 			break;
	 	}

 	molten::GLSLToSPIRVConverter glslConverter;
 	glslConverter.setGLSL(pshader);
 	bool wasConverted = glslConverter.convert(shaderStage, false, false);
 	if (wasConverted) { spirv = glslConverter.getSPIRV(); }
 	return wasConverted;
 }

#else   // not IOS OR macOS

#ifndef __ANDROID__
void init_resources(TBuiltInResource &Resources) {
    Resources.maxLights = 32;
    Resources.maxClipPlanes = 6;
    Resources.maxTextureUnits = 32;
    Resources.maxTextureCoords = 32;
    Resources.maxVertexAttribs = 64;
    Resources.maxVertexUniformComponents = 4096;
    Resources.maxVaryingFloats = 64;
    Resources.maxVertexTextureImageUnits = 32;
    Resources.maxCombinedTextureImageUnits = 80;
    Resources.maxTextureImageUnits = 32;
    Resources.maxFragmentUniformComponents = 4096;
    Resources.maxDrawBuffers = 32;
    Resources.maxVertexUniformVectors = 128;
    Resources.maxVaryingVectors = 8;
    Resources.maxFragmentUniformVectors = 16;
    Resources.maxVertexOutputVectors = 16;
    Resources.maxFragmentInputVectors = 15;
    Resources.minProgramTexelOffset = -8;
    Resources.maxProgramTexelOffset = 7;
    Resources.maxClipDistances = 8;
    Resources.maxComputeWorkGroupCountX = 65535;
    Resources.maxComputeWorkGroupCountY = 65535;
    Resources.maxComputeWorkGroupCountZ = 65535;
    Resources.maxComputeWorkGroupSizeX = 1024;
    Resources.maxComputeWorkGroupSizeY = 1024;
    Resources.maxComputeWorkGroupSizeZ = 64;
    Resources.maxComputeUniformComponents = 1024;
    Resources.maxComputeTextureImageUnits = 16;
    Resources.maxComputeImageUniforms = 8;
    Resources.maxComputeAtomicCounters = 8;
    Resources.maxComputeAtomicCounterBuffers = 1;
    Resources.maxVaryingComponents = 60;
    Resources.maxVertexOutputComponents = 64;
    Resources.maxGeometryInputComponents = 64;
    Resources.maxGeometryOutputComponents = 128;
    Resources.maxFragmentInputComponents = 128;
    Resources.maxImageUnits = 8;
    Resources.maxCombinedImageUnitsAndFragmentOutputs = 8;
    Resources.maxCombinedShaderOutputResources = 8;
    Resources.maxImageSamples = 0;
    Resources.maxVertexImageUniforms = 0;
    Resources.maxTessControlImageUniforms = 0;
    Resources.maxTessEvaluationImageUniforms = 0;
    Resources.maxGeometryImageUniforms = 0;
    Resources.maxFragmentImageUniforms = 8;
    Resources.maxCombinedImageUniforms = 8;
    Resources.maxGeometryTextureImageUnits = 16;
    Resources.maxGeometryOutputVertices = 256;
    Resources.maxGeometryTotalOutputComponents = 1024;
    Resources.maxGeometryUniformComponents = 1024;
    Resources.maxGeometryVaryingComponents = 64;
    Resources.maxTessControlInputComponents = 128;
    Resources.maxTessControlOutputComponents = 128;
    Resources.maxTessControlTextureImageUnits = 16;
    Resources.maxTessControlUniformComponents = 1024;
    Resources.maxTessControlTotalOutputComponents = 4096;
    Resources.maxTessEvaluationInputComponents = 128;
    Resources.maxTessEvaluationOutputComponents = 128;
    Resources.maxTessEvaluationTextureImageUnits = 16;
    Resources.maxTessEvaluationUniformComponents = 1024;
    Resources.maxTessPatchComponents = 120;
    Resources.maxPatchVertices = 32;
    Resources.maxTessGenLevel = 64;
    Resources.maxViewports = 16;
    Resources.maxVertexAtomicCounters = 0;
    Resources.maxTessControlAtomicCounters = 0;
    Resources.maxTessEvaluationAtomicCounters = 0;
    Resources.maxGeometryAtomicCounters = 0;
    Resources.maxFragmentAtomicCounters = 8;
    Resources.maxCombinedAtomicCounters = 8;
    Resources.maxAtomicCounterBindings = 1;
    Resources.maxVertexAtomicCounterBuffers = 0;
    Resources.maxTessControlAtomicCounterBuffers = 0;
    Resources.maxTessEvaluationAtomicCounterBuffers = 0;
    Resources.maxGeometryAtomicCounterBuffers = 0;
    Resources.maxFragmentAtomicCounterBuffers = 1;
    Resources.maxCombinedAtomicCounterBuffers = 1;
    Resources.maxAtomicCounterBufferSize = 16384;
    Resources.maxTransformFeedbackBuffers = 4;
    Resources.maxTransformFeedbackInterleavedComponents = 64;
    Resources.maxCullDistances = 8;
    Resources.maxCombinedClipAndCullDistances = 8;
    Resources.maxSamples = 4;
    Resources.limits.nonInductiveForLoops = 1;
    Resources.limits.whileLoops = 1;
    Resources.limits.doWhileLoops = 1;
    Resources.limits.generalUniformIndexing = 1;
    Resources.limits.generalAttributeMatrixVectorIndexing = 1;
    Resources.limits.generalVaryingIndexing = 1;
    Resources.limits.generalSamplerIndexing = 1;
    Resources.limits.generalVariableIndexing = 1;
    Resources.limits.generalConstantMatrixVectorIndexing = 1;
}

EShLanguage FindLanguage(const VkShaderStageFlagBits shader_type) {
    switch (shader_type) {
        case VK_SHADER_STAGE_VERTEX_BIT:
            return EShLangVertex;

        case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
            return EShLangTessControl;

        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
            return EShLangTessEvaluation;

        case VK_SHADER_STAGE_GEOMETRY_BIT:
            return EShLangGeometry;

        case VK_SHADER_STAGE_FRAGMENT_BIT:
            return EShLangFragment;

        case VK_SHADER_STAGE_COMPUTE_BIT:
            return EShLangCompute;

        default:
            return EShLangVertex;
    }
}
#endif

void init_glslang() {
#ifndef __ANDROID__
    glslang::InitializeProcess();
#endif
}

void finalize_glslang() {
#ifndef __ANDROID__
    glslang::FinalizeProcess();
#endif
}

#ifdef __ANDROID__
// Android specific helper functions for shaderc.
struct shader_type_mapping {
    VkShaderStageFlagBits vkshader_type;
    shaderc_shader_kind shaderc_type;
};
static const shader_type_mapping shader_map_table[] = {
    {VK_SHADER_STAGE_VERTEX_BIT, shaderc_glsl_vertex_shader},
    {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, shaderc_glsl_tess_control_shader},
    {VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, shaderc_glsl_tess_evaluation_shader},
    {VK_SHADER_STAGE_GEOMETRY_BIT, shaderc_glsl_geometry_shader},
    {VK_SHADER_STAGE_FRAGMENT_BIT, shaderc_glsl_fragment_shader},
    {VK_SHADER_STAGE_COMPUTE_BIT, shaderc_glsl_compute_shader},
};
shaderc_shader_kind MapShadercType(VkShaderStageFlagBits vkShader) {
    for (auto shader : shader_map_table) {
        if (shader.vkshader_type == vkShader) {
            return shader.shaderc_type;
        }
    }
    assert(false);
    return shaderc_glsl_infer_from_source;
}
#endif

//
// Compile a given string containing GLSL into SPV for use by VK
// Return value of false means an error was encountered.
//
bool GLSLtoSPV(const VkShaderStageFlagBits shader_type, const char *pshader, std::vector<unsigned int> &spirv) {
#ifndef __ANDROID__
    EShLanguage stage = FindLanguage(shader_type);
    glslang::TShader shader(stage);
    glslang::TProgram program;
    const char *shaderStrings[1];
    TBuiltInResource Resources;
    init_resources(Resources);

    // Enable SPIR-V and Vulkan rules when parsing GLSL
    EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

    shaderStrings[0] = pshader;
    shader.setStrings(shaderStrings, 1);

    if (!shader.parse(&Resources, 100, false, messages)) {
        puts(shader.getInfoLog());
        puts(shader.getInfoDebugLog());
        return false;  // something didn't work
    }

    program.addShader(&shader);

    //
    // Program-level processing...
    //

    if (!program.link(messages)) {
        puts(shader.getInfoLog());
        puts(shader.getInfoDebugLog());
        fflush(stdout);
        return false;
    }

    glslang::GlslangToSpv(*program.getIntermediate(stage), spirv);
#else
    // On Android, use shaderc instead.
    shaderc::Compiler compiler;
    shaderc::SpvCompilationResult module =
        compiler.CompileGlslToSpv(pshader, strlen(pshader), MapShadercType(shader_type), "shader");
    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        LOGE("Error: Id=%d, Msg=%s", module.GetCompilationStatus(), module.GetErrorMessage().c_str());
        return false;
    }
    spirv.assign(module.cbegin(), module.cend());
#endif
    return true;
}

#endif  // IOS or macOS

void wait_seconds(int seconds) {
#ifdef WIN32
    Sleep(seconds * 1000);
#elif defined(__ANDROID__)
    sleep(seconds);
#else
    sleep(seconds);
#endif
}

timestamp_t get_milliseconds() {
#ifdef WIN32
    LARGE_INTEGER frequency;
    BOOL useQPC = QueryPerformanceFrequency(&frequency);
    if (useQPC) {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        return (1000LL * now.QuadPart) / frequency.QuadPart;
    } else {
        return GetTickCount();
    }
#else
    struct timeval now;
    gettimeofday(&now, NULL);
    return (now.tv_usec / 1000) + (timestamp_t)now.tv_sec;
#endif
}

void print_UUID(uint8_t *pipelineCacheUUID) {
    for (int j = 0; j < VK_UUID_SIZE; ++j) {
        std::cout << std::setw(2) << (uint32_t)pipelineCacheUUID[j];
        if (j == 3 || j == 5 || j == 7 || j == 9) {
            std::cout << '-';
        }
    }
}
static bool optionMatch(const char *option, char *optionLine) {
    if (strncmp(option, optionLine, strlen(option)) == 0)
        return true;
    else
        return false;
}

void process_command_line_args(struct sample_info &info, int argc, char *argv[]) {
    int i, n;

    for (i = 1, n = 1; i < argc; i++) {
        if (optionMatch("--save-images", argv[i]))
            info.save_images = true;
        else if (optionMatch("--help", argv[i]) || optionMatch("-h", argv[i])) {
            printf("\nOther options:\n");
            printf(
                "\t--save-images\n"
                "\t\tSave tests images as ppm files in current working "
                "directory.\n");
            exit(0);
        } else {
            printf("\nUnrecognized option: %s\n", argv[i]);
            printf("\nUse --help or -h for option list.\n");
            exit(0);
        }

        /*
         * Since the above "consume" inputs, update argv
         * so that it contains the trimmed list of args for glutInit
         */

        argv[n] = argv[i];
        n++;
    }
}

void write_ppm(struct sample_info &info, const char *basename) {
    string filename;
    int x, y;
    VkResult res;

    VkImageCreateInfo image_create_info = {};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.pNext = NULL;
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.format = info.format;
    image_create_info.extent.width = info.width;
    image_create_info.extent.height = info.height;
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling = VK_IMAGE_TILING_LINEAR;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_create_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_create_info.queueFamilyIndexCount = 0;
    image_create_info.pQueueFamilyIndices = NULL;
    image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_create_info.flags = 0;

    VkMemoryAllocateInfo mem_alloc = {};
    mem_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mem_alloc.pNext = NULL;
    mem_alloc.allocationSize = 0;
    mem_alloc.memoryTypeIndex = 0;

    VkImage mappableImage;
    VkDeviceMemory mappableMemory;

    /* Create a mappable image */
    res = vkCreateImage(info.device, &image_create_info, NULL, &mappableImage);
    assert(res == VK_SUCCESS);

    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(info.device, mappableImage, &mem_reqs);

    mem_alloc.allocationSize = mem_reqs.size;

    /* Find the memory type that is host mappable */
    bool pass = memory_type_from_properties(info, mem_reqs.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            &mem_alloc.memoryTypeIndex);
    assert(pass && "No mappable, coherent memory");

    /* allocate memory */
    res = vkAllocateMemory(info.device, &mem_alloc, NULL, &(mappableMemory));
    assert(res == VK_SUCCESS);

    /* bind memory */
    res = vkBindImageMemory(info.device, mappableImage, mappableMemory, 0);
    assert(res == VK_SUCCESS);

    VkCommandBufferBeginInfo cmd_buf_info = {};
    cmd_buf_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_buf_info.pNext = NULL;
    cmd_buf_info.flags = 0;
    cmd_buf_info.pInheritanceInfo = NULL;

    res = vkBeginCommandBuffer(info.cmd, &cmd_buf_info);
    set_image_layout(info, mappableImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    set_image_layout(info, info.buffers[info.current_buffer].image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageCopy copy_region;
    copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.srcSubresource.mipLevel = 0;
    copy_region.srcSubresource.baseArrayLayer = 0;
    copy_region.srcSubresource.layerCount = 1;
    copy_region.srcOffset.x = 0;
    copy_region.srcOffset.y = 0;
    copy_region.srcOffset.z = 0;
    copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.dstSubresource.mipLevel = 0;
    copy_region.dstSubresource.baseArrayLayer = 0;
    copy_region.dstSubresource.layerCount = 1;
    copy_region.dstOffset.x = 0;
    copy_region.dstOffset.y = 0;
    copy_region.dstOffset.z = 0;
    copy_region.extent.width = info.width;
    copy_region.extent.height = info.height;
    copy_region.extent.depth = 1;

    /* Put the copy command into the command buffer */
    vkCmdCopyImage(info.cmd, info.buffers[info.current_buffer].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mappableImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    set_image_layout(info, mappableImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);

    res = vkEndCommandBuffer(info.cmd);
    assert(res == VK_SUCCESS);
    const VkCommandBuffer cmd_bufs[] = {info.cmd};
    VkFenceCreateInfo fenceInfo;
    VkFence cmdFence;
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = NULL;
    fenceInfo.flags = 0;
    vkCreateFence(info.device, &fenceInfo, NULL, &cmdFence);

    VkSubmitInfo submit_info[1] = {};
    submit_info[0].pNext = NULL;
    submit_info[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info[0].waitSemaphoreCount = 0;
    submit_info[0].pWaitSemaphores = NULL;
    submit_info[0].pWaitDstStageMask = NULL;
    submit_info[0].commandBufferCount = 1;
    submit_info[0].pCommandBuffers = cmd_bufs;
    submit_info[0].signalSemaphoreCount = 0;
    submit_info[0].pSignalSemaphores = NULL;

    /* Queue the command buffer for execution */
    res = vkQueueSubmit(info.graphics_queue, 1, submit_info, cmdFence);
    assert(res == VK_SUCCESS);

    /* Make sure command buffer is finished before mapping */
    do {
        res = vkWaitForFences(info.device, 1, &cmdFence, VK_TRUE, FENCE_TIMEOUT);
    } while (res == VK_TIMEOUT);
    assert(res == VK_SUCCESS);

    vkDestroyFence(info.device, cmdFence, NULL);

    filename.append(basename);
    filename.append(".ppm");

    VkImageSubresource subres = {};
    subres.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.mipLevel = 0;
    subres.arrayLayer = 0;
    VkSubresourceLayout sr_layout;
    vkGetImageSubresourceLayout(info.device, mappableImage, &subres, &sr_layout);

    char *ptr;
    res = vkMapMemory(info.device, mappableMemory, 0, mem_reqs.size, 0, (void **)&ptr);
    assert(res == VK_SUCCESS);

    ptr += sr_layout.offset;
    ofstream file(filename.c_str(), ios::binary);

    file << "P6\n";
    file << info.width << " ";
    file << info.height << "\n";
    file << 255 << "\n";

    for (y = 0; y < info.height; y++) {
        const int *row = (const int *)ptr;
        int swapped;

        if (info.format == VK_FORMAT_B8G8R8A8_UNORM || info.format == VK_FORMAT_B8G8R8A8_SRGB) {
            for (x = 0; x < info.width; x++) {
                swapped = (*row & 0xff00ff00) | (*row & 0x000000ff) << 16 | (*row & 0x00ff0000) >> 16;
                file.write((char *)&swapped, 3);
                row++;
            }
        } else if (info.format == VK_FORMAT_R8G8B8A8_UNORM) {
            for (x = 0; x < info.width; x++) {
                file.write((char *)row, 3);
                row++;
            }
        } else {
            printf("Unrecognized image format - will not write image files");
            break;
        }

        ptr += sr_layout.rowPitch;
    }

    file.close();
    vkUnmapMemory(info.device, mappableMemory);
    vkDestroyImage(info.device, mappableImage, NULL);
    vkFreeMemory(info.device, mappableMemory, NULL);
}

std::string get_file_directory() {
#ifndef __ANDROID__
    return "";
#else
    assert(Android_application != nullptr);
    return Android_application->activity->externalDataPath;
#endif
}

#ifdef __ANDROID__
//
// Android specific helper functions.
//

// Helpder class to forward the cout/cerr output to logcat derived from:
// http://stackoverflow.com/questions/8870174/is-stdcout-usable-in-android-ndk
class AndroidBuffer : public std::streambuf {
   public:
    AndroidBuffer(android_LogPriority priority) {
        priority_ = priority;
        this->setp(buffer_, buffer_ + kBufferSize - 1);
    }

   private:
    static const int32_t kBufferSize = 128;
    int32_t overflow(int32_t c) {
        if (c == traits_type::eof()) {
            *this->pptr() = traits_type::to_char_type(c);
            this->sbumpc();
        }
        return this->sync() ? traits_type::eof() : traits_type::not_eof(c);
    }

    int32_t sync() {
        int32_t rc = 0;
        if (this->pbase() != this->pptr()) {
            char writebuf[kBufferSize + 1];
            memcpy(writebuf, this->pbase(), this->pptr() - this->pbase());
            writebuf[this->pptr() - this->pbase()] = '\0';

            rc = __android_log_write(priority_, "std", writebuf) > 0;
            this->setp(buffer_, buffer_ + kBufferSize - 1);
        }
        return rc;
    }

    android_LogPriority priority_ = ANDROID_LOG_INFO;
    char buffer_[kBufferSize];
};

void Android_handle_cmd(android_app *app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            // The window is being shown, get it ready.
            sample_main(0, nullptr);
            LOGI("\n");
            LOGI("=================================================");
            LOGI("          The sample ran successfully!!");
            LOGI("=================================================");
            LOGI("\n");
            break;
        case APP_CMD_TERM_WINDOW:
            // The window is being hidden or closed, clean it up.
            break;
        default:
            LOGI("event not handled: %d", cmd);
    }
}

bool Android_process_command() {
    assert(Android_application != nullptr);
    int events;
    android_poll_source *source;
    // Poll all pending events.
    if (ALooper_pollAll(0, NULL, &events, (void **)&source) >= 0) {
        // Process each polled events
        if (source != NULL) source->process(Android_application, source);
    }
    return Android_application->destroyRequested;
}

void android_main(struct android_app *app) {
    // Set static variables.
    Android_application = app;
    // Set the callback to process system events
    app->onAppCmd = Android_handle_cmd;

    // Forward cout/cerr to logcat.
    std::cout.rdbuf(new AndroidBuffer(ANDROID_LOG_INFO));
    std::cerr.rdbuf(new AndroidBuffer(ANDROID_LOG_ERROR));

    // Main loop
    do {
        Android_process_command();
    }  // Check if system requested to quit the application
    while (app->destroyRequested == 0);

    return;
}

ANativeWindow *AndroidGetApplicationWindow() {
    assert(Android_application != nullptr);
    return Android_application->window;
}

bool AndroidFillShaderMap(const char *path, std::unordered_map<std::string, std::string> *map_shaders) {
    assert(Android_application != nullptr);
    auto directory = AAssetManager_openDir(Android_application->activity->assetManager, path);

    const char *name = nullptr;
    while (1) {
        name = AAssetDir_getNextFileName(directory);
        if (name == nullptr) {
            break;
        }

        std::string file_name = name;
        if (file_name.find(".frag") != std::string::npos || file_name.find(".vert") != std::string::npos) {
            // Add path to the filename.
            file_name = std::string(path) + "/" + file_name;
            std::string shader;
            if (!AndroidLoadFile(file_name.c_str(), &shader)) {
                continue;
            }
            // Remove \n to make the lookup more robust.
            while (1) {
                auto ret_pos = shader.find("\n");
                if (ret_pos == std::string::npos) {
                    break;
                }
                shader.erase(ret_pos, 1);
            }

            auto pos = file_name.find_last_of(".");
            if (pos == std::string::npos) {
                // Invalid file nmae.
                continue;
            }
            // Generate filename for SPIRV binary.
            std::string spirv_name = file_name.replace(pos, 1, "-") + ".spirv";
            // Store the SPIRV file name with GLSL contents as a key.
            // The file contents can be long, but as we are using unordered map, it wouldn't take
            // much storage space.
            // Put the file into the map.
            (*map_shaders)[shader] = spirv_name;
        }
    };

    AAssetDir_close(directory);
    return true;
}

bool AndroidLoadFile(const char *filePath, std::string *data) {
    assert(Android_application != nullptr);
    AAsset *file = AAssetManager_open(Android_application->activity->assetManager, filePath, AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    LOGI("Loaded file:%s size:%zu", filePath, fileLength);
    if (fileLength == 0) {
        return false;
    }
    data->resize(fileLength);
    AAsset_read(file, &(*data)[0], fileLength);
    return true;
}

void AndroidGetWindowSize(int32_t *width, int32_t *height) {
    // On Android, retrieve the window size from the native window.
    assert(Android_application != nullptr);
    *width = ANativeWindow_getWidth(Android_application->window);
    *height = ANativeWindow_getHeight(Android_application->window);
}

// Android fopen stub described at
// http://www.50ply.com/blog/2013/01/19/loading-compressed-android-assets-with-file-pointer/#comment-1850768990
static int android_read(void *cookie, char *buf, int size) { return AAsset_read((AAsset *)cookie, buf, size); }

static int android_write(void *cookie, const char *buf, int size) {
    return EACCES;  // can't provide write access to the apk
}

static fpos_t android_seek(void *cookie, fpos_t offset, int whence) { return AAsset_seek((AAsset *)cookie, offset, whence); }

static int android_close(void *cookie) {
    AAsset_close((AAsset *)cookie);
    return 0;
}

FILE *AndroidFopen(const char *fname, const char *mode) {
    if (mode[0] == 'w') {
        return NULL;
    }

    assert(Android_application != nullptr);
    AAsset *asset = AAssetManager_open(Android_application->activity->assetManager, fname, 0);
    if (!asset) {
        return NULL;
    }

    return funopen(asset, android_read, android_write, android_seek, android_close);
}
#endif
