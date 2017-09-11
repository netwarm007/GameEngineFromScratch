/*
 * Vulkan Samples
 *
 * Copyright (C) 2016 Valve Corporation
 * Copyright (C) 2016 LunarG, Inc.
 * Copyright (C) 2016 Google, Inc.
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
VULKAN_SAMPLE_SHORT_DESCRIPTION
Create and use a pipeline cache accross runs.
*/

#include <util_init.hpp>
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include "cube_data.h"

// This sample tries to save and reuse pipeline cache data between runs
// On first run, no cache will be found, it will be created and saved
// to disk. On later runs, the cache should be found, loaded, and used.
// Hopefully a speedup will observed.  In the future, the pipeline could
// be complicated a bit, to show a greater cache benefit.  Also, two
// pipelines could be created and merged.

const char *vertShaderText =
    "#version 400\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "#extension GL_ARB_shading_language_420pack : enable\n"
    "layout (std140, binding = 0) uniform buf {\n"
    "        mat4 mvp;\n"
    "} ubuf;\n"
    "layout (location = 0) in vec4 pos;\n"
    "layout (location = 1) in vec2 inTexCoords;\n"
    "layout (location = 0) out vec2 texcoord;\n"
    "void main() {\n"
    "   texcoord = inTexCoords;\n"
    "   gl_Position = ubuf.mvp * pos;\n"
    "}\n";

const char *fragShaderText =
    "#version 400\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "#extension GL_ARB_shading_language_420pack : enable\n"
    "layout (binding = 1) uniform sampler2D tex;\n"
    "layout (location = 0) in vec2 texcoord;\n"
    "layout (location = 0) out vec4 outColor;\n"
    "void main() {\n"
    "   outColor = textureLod(tex, texcoord, 0.0);\n"
    "}\n";

int sample_main(int argc, char *argv[]) {
    VkResult U_ASSERT_ONLY res;
    struct sample_info info = {};
    char sample_title[] = "Pipeline Cache";
    const bool depthPresent = true;

    process_command_line_args(info, argc, argv);
    init_global_layer_properties(info);
    init_instance_extension_names(info);
    init_device_extension_names(info);
    init_instance(info, sample_title);
    init_enumerate_device(info);
    init_window_size(info, 500, 500);
    init_connection(info);
    init_window(info);
    init_swapchain_extension(info);
    init_device(info);
    init_command_pool(info);
    init_command_buffer(info);
    execute_begin_command_buffer(info);
    init_device_queue(info);
    init_swap_chain(info);
    init_depth_buffer(info);
    init_texture(info, "blue.ppm");
    init_uniform_buffer(info);
    init_descriptor_and_pipeline_layouts(info, true);
    init_renderpass(info, depthPresent);
    init_shaders(info, vertShaderText, fragShaderText);
    init_framebuffers(info, depthPresent);
    init_vertex_buffer(info, g_vb_texture_Data, sizeof(g_vb_texture_Data), sizeof(g_vb_texture_Data[0]), true);
    init_descriptor_pool(info, true);
    init_descriptor_set(info, true);

    /* VULKAN_KEY_START */

    // Check disk for existing cache data
    size_t startCacheSize = 0;
    void *startCacheData = nullptr;

    std::string directoryName = get_file_directory();
    std::string readFileName = directoryName + "pipeline_cache_data.bin";
    FILE *pReadFile = fopen(readFileName.c_str(), "rb");

    if (pReadFile) {
        // Determine cache size
        fseek(pReadFile, 0, SEEK_END);
        startCacheSize = ftell(pReadFile);
        rewind(pReadFile);

        // Allocate memory to hold the initial cache data
        startCacheData = (char *)malloc(sizeof(char) * startCacheSize);
        if (startCacheData == nullptr) {
            fputs("Memory error", stderr);
            exit(EXIT_FAILURE);
        }

        // Read the data into our buffer
        size_t result = fread(startCacheData, 1, startCacheSize, pReadFile);
        if (result != startCacheSize) {
            fputs("Reading error", stderr);
            free(startCacheData);
            exit(EXIT_FAILURE);
        }

        // Clean up and print results
        fclose(pReadFile);
        printf("  Pipeline cache HIT!\n");
        printf("  cacheData loaded from %s\n", readFileName.c_str());

    } else {
        // No cache found on disk
        printf("  Pipeline cache miss!\n");
    }

    if (startCacheData != nullptr) {
        // clang-format off
        //
        // Check for cache validity
        //
        // TODO: Update this as the spec evolves. The fields are not defined by the header.
        //
        // The code below supports SDK 0.10 Vulkan spec, which contains the following table:
        //
        // Offset	 Size            Meaning
        // ------    ------------    ------------------------------------------------------------------
        //      0               4    a device ID equal to VkPhysicalDeviceProperties::DeviceId written
        //                           as a stream of bytes, with the least significant byte first
        //
        //      4    VK_UUID_SIZE    a pipeline cache ID equal to VkPhysicalDeviceProperties::pipelineCacheUUID
        //
        //
        // The code must be updated for latest Vulkan spec, which contains the following table:
        //
        // Offset	 Size            Meaning
        // ------    ------------    ------------------------------------------------------------------
        //      0               4    length in bytes of the entire pipeline cache header written as a
        //                           stream of bytes, with the least significant byte first
        //      4               4    a VkPipelineCacheHeaderVersion value written as a stream of bytes,
        //                           with the least significant byte first
        //      8               4    a vendor ID equal to VkPhysicalDeviceProperties::vendorID written
        //                           as a stream of bytes, with the least significant byte first
        //     12               4    a device ID equal to VkPhysicalDeviceProperties::deviceID written
        //                           as a stream of bytes, with the least significant byte first
        //     16    VK_UUID_SIZE    a pipeline cache ID equal to VkPhysicalDeviceProperties::pipelineCacheUUID
        //
        // clang-format on
        uint32_t headerLength = 0;
        uint32_t cacheHeaderVersion = 0;
        uint32_t vendorID = 0;
        uint32_t deviceID = 0;
        uint8_t pipelineCacheUUID[VK_UUID_SIZE] = {};

        memcpy(&headerLength, (uint8_t *)startCacheData + 0, 4);
        memcpy(&cacheHeaderVersion, (uint8_t *)startCacheData + 4, 4);
        memcpy(&vendorID, (uint8_t *)startCacheData + 8, 4);
        memcpy(&deviceID, (uint8_t *)startCacheData + 12, 4);
        memcpy(pipelineCacheUUID, (uint8_t *)startCacheData + 16, VK_UUID_SIZE);

        // Check each field and report bad values before freeing existing cache
        bool badCache = false;

        if (headerLength <= 0) {
            badCache = true;
            printf("  Bad header length in %s.\n", readFileName.c_str());
            printf("    Cache contains: 0x%.8x\n", headerLength);
        }

        if (cacheHeaderVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE) {
            badCache = true;
            printf("  Unsupported cache header version in %s.\n", readFileName.c_str());
            printf("    Cache contains: 0x%.8x\n", cacheHeaderVersion);
        }

        if (vendorID != info.gpu_props.vendorID) {
            badCache = true;
            printf("  Vendor ID mismatch in %s.\n", readFileName.c_str());
            printf("    Cache contains: 0x%.8x\n", vendorID);
            printf("    Driver expects: 0x%.8x\n", info.gpu_props.vendorID);
        }

        if (deviceID != info.gpu_props.deviceID) {
            badCache = true;
            printf("  Device ID mismatch in %s.\n", readFileName.c_str());
            printf("    Cache contains: 0x%.8x\n", deviceID);
            printf("    Driver expects: 0x%.8x\n", info.gpu_props.deviceID);
        }

        if (memcmp(pipelineCacheUUID, info.gpu_props.pipelineCacheUUID, sizeof(pipelineCacheUUID)) != 0) {
            badCache = true;
            printf("  UUID mismatch in %s.\n", readFileName.c_str());
            printf("    Cache contains: ");
            print_UUID(pipelineCacheUUID);
            printf("\n");
            printf("    Driver expects: ");
            print_UUID(info.gpu_props.pipelineCacheUUID);
            printf("\n");
        }

        if (badCache) {
            // Don't submit initial cache data if any version info is incorrect
            free(startCacheData);
            startCacheSize = 0;
            startCacheData = nullptr;

            // And clear out the old cache file for use in next run
            printf("  Deleting cache entry %s to repopulate.\n", readFileName.c_str());
            if (remove(readFileName.c_str()) != 0) {
                fputs("Reading error", stderr);
                exit(EXIT_FAILURE);
            }
        }
    }

    // Feed the initial cache data into pipeline creation
    VkPipelineCacheCreateInfo pipelineCache;
    pipelineCache.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pipelineCache.pNext = NULL;
    pipelineCache.initialDataSize = startCacheSize;
    pipelineCache.pInitialData = startCacheData;
    pipelineCache.flags = 0;
    res = vkCreatePipelineCache(info.device, &pipelineCache, nullptr, &info.pipelineCache);
    assert(res == VK_SUCCESS);

    // Free our initialData now that pipeline has been created
    free(startCacheData);

    // Time (roughly) taken to create the graphics pipeline
    timestamp_t start = get_milliseconds();
    init_pipeline(info, depthPresent);
    timestamp_t elapsed = get_milliseconds() - start;
    printf("  vkCreateGraphicsPipeline time: %0.f ms\n", (double)elapsed);

    // Begin standard draw stuff

    init_presentable_image(info);
    VkClearValue clear_values[2];
    init_clear_color_and_depth(info, clear_values);
    VkRenderPassBeginInfo rp_begin;
    init_render_pass_begin_info(info, rp_begin);
    rp_begin.clearValueCount = 2;
    rp_begin.pClearValues = clear_values;
    vkCmdBeginRenderPass(info.cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline);
    vkCmdBindDescriptorSets(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline_layout, 0, NUM_DESCRIPTOR_SETS,
                            info.desc_set.data(), 0, NULL);
    const VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(info.cmd, 0, 1, &info.vertex_buffer.buf, offsets);
    init_viewports(info);
    init_scissors(info);
    vkCmdDraw(info.cmd, 12 * 3, 1, 0, 0);
    vkCmdEndRenderPass(info.cmd);
    res = vkEndCommandBuffer(info.cmd);
    assert(res == VK_SUCCESS);
    VkFence drawFence = {};
    init_fence(info, drawFence);
    VkPipelineStageFlags pipe_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info = {};
    init_submit_info(info, submit_info, pipe_stage_flags);
    /* Queue the command buffer for execution */
    res = vkQueueSubmit(info.graphics_queue, 1, &submit_info, drawFence);
    assert(res == VK_SUCCESS);
    /* Now present the image in the window */
    VkPresentInfoKHR present = {};
    init_present_info(info, present);
    /* Make sure command buffer is finished before presenting */
    do {
        res = vkWaitForFences(info.device, 1, &drawFence, VK_TRUE, FENCE_TIMEOUT);
    } while (res == VK_TIMEOUT);
    assert(res == VK_SUCCESS);
    res = vkQueuePresentKHR(info.present_queue, &present);
    assert(res == VK_SUCCESS);
    wait_seconds(1);
    if (info.save_images) write_ppm(info, "pipeline_cache");

    // End standard draw stuff

    if (startCacheData) {
        // TODO: Create another pipeline, preferably different from the first
        // one and merge it here.  Then store the merged one.
    }

    // Store away the cache that we've populated.  This could conceivably happen
    // earlier, depends on when the pipeline cache stops being populated
    // internally.
    size_t endCacheSize = 0;
    void *endCacheData = nullptr;

    // Call with nullptr to get cache size
    res = vkGetPipelineCacheData(info.device, info.pipelineCache, &endCacheSize, nullptr);
    assert(res == VK_SUCCESS);

    // Allocate memory to hold the populated cache data
    endCacheData = (char *)malloc(sizeof(char) * endCacheSize);
    if (!endCacheData) {
        fputs("Memory error", stderr);
        exit(EXIT_FAILURE);
    }

    // Call again with pointer to buffer
    res = vkGetPipelineCacheData(info.device, info.pipelineCache, &endCacheSize, endCacheData);
    assert(res == VK_SUCCESS);

    // Write the file to disk, overwriting whatever was there
    FILE *pWriteFile;
    std::string writeFileName = directoryName + "pipeline_cache_data.bin";
    pWriteFile = fopen(writeFileName.c_str(), "wb");
    if (pWriteFile) {
        fwrite(endCacheData, sizeof(char), endCacheSize, pWriteFile);
        fclose(pWriteFile);
        printf("  cacheData written to %s\n", writeFileName.c_str());
    } else {
        // Something bad happened
        printf("  Unable to write cache data to disk!\n");
    }

    /* VULKAN_KEY_END */

    vkDestroyFence(info.device, drawFence, NULL);
    vkDestroySemaphore(info.device, info.imageAcquiredSemaphore, NULL);
    destroy_pipeline(info);
    destroy_pipeline_cache(info);
    destroy_textures(info);
    destroy_descriptor_pool(info);
    destroy_vertex_buffer(info);
    destroy_framebuffers(info);
    destroy_shaders(info);
    destroy_renderpass(info);
    destroy_descriptor_and_pipeline_layouts(info);
    destroy_uniform_buffer(info);
    destroy_depth_buffer(info);
    destroy_swap_chain(info);
    destroy_command_buffer(info);
    destroy_command_pool(info);
    destroy_device(info);
    destroy_window(info);
    destroy_instance(info);
    return 0;
}
