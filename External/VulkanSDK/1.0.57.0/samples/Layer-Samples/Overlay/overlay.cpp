/*
 * Vulkan
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 * Copyright (C) 2015-2016 Google Inc.
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
 *
 * Author: Chris Forbes <chrisforbes@google.com>
 */
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unordered_map>
#include <vector>
#include "util.hpp"
#include <vk_loader_platform.h>
#include <vulkan/vulkan.h>
#include <vk_dispatch_table_helper.h>
#include <vulkan/vk_layer.h>
#include "vk_layer_data.h"
#include "vk_layer_table.h"
#include "vk_layer_extension_utils.h"

#define STB_TRUETYPE_IMPLEMENTATION
//#define STBTT_STATIC
#include "stb_truetype.h"

struct vertex {
    float x, y, u, v;
};

#define MAX_TEXT_VERTICES 16384
#define FONT_SIZE_PIXELS 18
#define FONT_ATLAS_SIZE 512

struct WsiImageData {
    VkImage image;
    VkImageView view;
    VkFramebuffer framebuffer;
    VkCommandBuffer cmd;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    int numVertices;
    VkDeviceSize vertexBufferSize;

    void Cleanup(VkDevice dev);
};

struct SwapChainData {
    unsigned width, height;
    VkFormat format;

    VkRenderPass render_pass;
    VkPipeline pipeline;

    std::vector<WsiImageData *> presentableImages;

    void Cleanup(VkDevice dev);
};

struct layer_data {
    VkLayerDispatchTable *device_dispatch_table;
    VkLayerInstanceDispatchTable *instance_dispatch_table;

    PFN_vkCreateSwapchainKHR pfnCreateSwapchainKHR;
    PFN_vkGetSwapchainImagesKHR pfnGetSwapchainImagesKHR;
    PFN_vkQueuePresentKHR pfnQueuePresentKHR;
    PFN_vkDestroySwapchainKHR pfnDestroySwapchainKHR;

    VkPhysicalDevice gpu;
    VkDevice dev;
    uint32_t graphicsQueueFamilyIndex;

    PFN_vkSetDeviceLoaderData pfn_dev_init;
    std::unordered_map<VkSwapchainKHR, SwapChainData *> *swapChains;
    VkCommandPool pool;

    VkPipelineCache pipelineCache;

    VkShaderModule vsShaderModule;
    VkShaderModule fsShaderModule;

    VkImage fontGlyphsImage;
    VkImageView fontGlyphsImageView;
    VkDeviceMemory fontGlyphsMemory;
    stbtt_bakedchar glyphs[96];
    VkCommandBuffer fontUploadCmdBuffer;
    bool fontUploadComplete;

    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkDescriptorPool desc_pool;
    VkDescriptorSet desc_set;
    VkSampler sampler;
    VkFence fence;

    int frame;
    int cmdBuffersThisFrame;

    void Cleanup();
};

static std::unordered_map<void *, layer_data *> layer_data_map;

template layer_data *GetLayerDataPtr<layer_data>(void *data_key, std::unordered_map<void *, layer_data *> &data_map);

// static LOADER_PLATFORM_THREAD_ONCE_DECLARATION(g_initOnce);
// TODO : This can be much smarter, using separate locks for separate global
// data
static int globalLockInitialized = 0;
static loader_platform_thread_mutex globalLock;

static void init_overlay(layer_data *my_data) {
    if (!globalLockInitialized) {
        // TODO/TBD: Need to delete this mutex sometime.  How???  One
        // suggestion is to call this during vkCreateInstance(), and then we
        // can clean it up during vkDestroyInstance().  However, that requires
        // that the layer have per-instance locks.  We need to come back and
        // address this soon.
        loader_platform_thread_create_mutex(&globalLock);
        globalLockInitialized = 1;
    }
}

static bool get_file_contents(char const *filename, std::vector<unsigned char> &vec) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
#ifdef OVERLAY_DEBUG
        printf("Failed to open `%s`\n", filename);
#endif
        return false;
    }

    fseek(f, 0, SEEK_END);
    auto length = ftell(f);

    fseek(f, 0, SEEK_SET);

    vec.resize(length);
    if ((uint32_t)length != fread(&vec[0], 1, length, f)) {
#ifdef OVERLAY_DEBUG
        printf("Short read `%s`\n", filename);
#endif
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

static bool compile_shader(VkDevice device, char const *filename, VkShaderModule *module) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);

    std::vector<unsigned char> bytecode;
    if (!get_file_contents(filename, bytecode)) {
        return false;
    }

    VkResult U_ASSERT_ONLY res;

    VkShaderModuleCreateInfo smci;
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.pNext = nullptr;
    smci.codeSize = bytecode.size();
    smci.pCode = (uint32_t const *)&bytecode[0];
    smci.flags = 0;

    res = my_data->device_dispatch_table->CreateShaderModule(device, &smci, nullptr, module);
    assert(!res);

#ifdef OVERLAY_DEBUG
    printf("Compiled shader for overlay: `%s`\n", filename);
#endif

    return true;
}

static uint32_t choose_memory_type(VkPhysicalDevice gpu, uint32_t typeBits, VkMemoryPropertyFlags properties) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(gpu), layer_data_map);

    VkPhysicalDeviceMemoryProperties props;
    my_data->instance_dispatch_table->GetPhysicalDeviceMemoryProperties(gpu, &props);

    for (auto i = 0u; i < props.memoryTypeCount; i++) {
        if ((1 << i) & typeBits) {
            if ((props.memoryTypes[i].propertyFlags & properties) == properties) return i;
        }
    }

    assert(!"Failed to choose memory type");
    return 0;
}

static int fill_vertex_buffer(layer_data *data, vertex *vertices, int index) {
    char str[1024];
    sprintf(str,
            "Vulkan Overlay Example\nWSI Image Index: %d\nFrame: "
            "%d\nCmdBuffers: %d",
            index, data->frame++, data->cmdBuffersThisFrame);
    float x = 0;
    float y = 16;

    vertex *v = vertices;

    for (char const *p = str; *p; p++) {
        if (*p == '\n') {
            y += 16;
            x = 0;
        } else {
            stbtt_aligned_quad q;
            stbtt_GetBakedQuad(data->glyphs, FONT_ATLAS_SIZE, FONT_ATLAS_SIZE, *p - 32, &x, &y, &q, 1);

            v[0].x = q.x0;
            v[0].y = q.y0;
            v[0].u = q.s0;
            v[0].v = q.t0;
            v[1].x = q.x1;
            v[1].y = q.y0;
            v[1].u = q.s1;
            v[1].v = q.t0;
            v[2].x = q.x0;
            v[2].y = q.y1;
            v[2].u = q.s0;
            v[2].v = q.t1;

            v[3] = v[1];
            v[4].x = q.x1;
            v[4].y = q.y1;
            v[4].u = q.s1;
            v[4].v = q.t1;
            v[5] = v[2];

            v += 6;
        }
    }

    return (int)(v - vertices);
}

static void after_device_create(VkPhysicalDevice gpu, VkDevice device, layer_data *data) {
    VkResult U_ASSERT_ONLY err;

    data->gpu = gpu;
    data->dev = device;
    data->frame = 0;
    data->cmdBuffersThisFrame = 0;

    VkLayerDispatchTable *pTable = data->device_dispatch_table;

    /* Get our WSI hooks in. */
    data->pfnCreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)pTable->GetDeviceProcAddr(device, "vkCreateSwapchainKHR");
    data->pfnGetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)pTable->GetDeviceProcAddr(device, "vkGetSwapchainImagesKHR");
    data->pfnQueuePresentKHR = (PFN_vkQueuePresentKHR)pTable->GetDeviceProcAddr(device, "vkQueuePresentKHR");
    data->pfnDestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)pTable->GetDeviceProcAddr(device, "vkDestroySwapchainKHR");
    data->swapChains = new std::unordered_map<VkSwapchainKHR, SwapChainData *>;

    /* Command pool */
    VkCommandPoolCreateInfo cpci;
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.pNext = nullptr;
    cpci.queueFamilyIndex = data->graphicsQueueFamilyIndex;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    err = pTable->CreateCommandPool(device, &cpci, nullptr, &data->pool);
    assert(!err);

    /* Create the objects we need */

    /* Compile the shaders */
    compile_shader(device, VULKAN_SAMPLES_BASE_DIR "/Layer-Samples/data/overlay-vert.spv", &data->vsShaderModule);
    compile_shader(device, VULKAN_SAMPLES_BASE_DIR "/Layer-Samples/data/overlay-frag.spv", &data->fsShaderModule);

    /* Upload the font bitmap */
    VkImageCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8_UNORM;
    ici.extent.width = FONT_ATLAS_SIZE;
    ici.extent.height = FONT_ATLAS_SIZE;
    ici.extent.depth = 1;
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_LINEAR;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;

    err = pTable->CreateImage(device, &ici, nullptr, &data->fontGlyphsImage);
    assert(!err);

    VkMemoryRequirements mem_reqs;
    pTable->GetImageMemoryRequirements(device, data->fontGlyphsImage, &mem_reqs);

    VkMemoryAllocateInfo mem_alloc;
    memset(&mem_alloc, 0, sizeof(mem_alloc));
    mem_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mem_alloc.allocationSize = mem_reqs.size;
    mem_alloc.memoryTypeIndex = choose_memory_type(gpu, mem_reqs.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    err = pTable->AllocateMemory(device, &mem_alloc, nullptr, &data->fontGlyphsMemory);
    assert(!err);
    err = pTable->BindImageMemory(device, data->fontGlyphsImage, data->fontGlyphsMemory, 0);
    assert(!err);

    VkImageSubresource subres;
    subres.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.mipLevel = 0;
    subres.arrayLayer = 0;
    VkSubresourceLayout layout;
    void *bits;

    pTable->GetImageSubresourceLayout(device, data->fontGlyphsImage, &subres, &layout);

    /* ensure we can directly upload into this layout */
    assert(!layout.offset);
    assert(layout.size >= FONT_ATLAS_SIZE * FONT_ATLAS_SIZE);
    assert(layout.rowPitch == FONT_ATLAS_SIZE);

    err = pTable->MapMemory(device, data->fontGlyphsMemory, 0, VK_WHOLE_SIZE, 0, &bits);
    assert(!err);

    /* Load the font glyphs directly into the mapped buffer */
    std::vector<unsigned char> fontData;
    get_file_contents(VULKAN_SAMPLES_BASE_DIR "/Layer-Samples/data/FreeSans.ttf", fontData);
    stbtt_BakeFontBitmap(&fontData[0], 0, FONT_SIZE_PIXELS, (unsigned char *)bits, FONT_ATLAS_SIZE, FONT_ATLAS_SIZE, 32, 96,
                         data->glyphs);

    pTable->UnmapMemory(device, data->fontGlyphsMemory);

    VkImageViewCreateInfo ivci;
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.pNext = nullptr;
    ivci.format = ici.format;
    ivci.components.r = VK_COMPONENT_SWIZZLE_R;
    ivci.components.g = VK_COMPONENT_SWIZZLE_G;
    ivci.components.b = VK_COMPONENT_SWIZZLE_B;
    ivci.components.a = VK_COMPONENT_SWIZZLE_A;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.baseMipLevel = 0;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.baseArrayLayer = 0;
    ivci.subresourceRange.layerCount = 1;
    ivci.image = data->fontGlyphsImage;
    ivci.flags = 0;

    err = pTable->CreateImageView(device, &ivci, nullptr, &data->fontGlyphsImageView);
    assert(!err);

    /* transition from undefined layout to shader readonly so we can use it.
     * requires a command buffer. */
    VkCommandBufferAllocateInfo cbai;
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.pNext = nullptr;
    cbai.commandPool = data->pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    err = pTable->AllocateCommandBuffers(device, &cbai, &cmd);
    assert(!err);

    /* We have just created a dispatchable object, but the dispatch table has
     * not been placed in the object yet.
     * When a "normal" application creates a command buffer,
     * the dispatch table is installed by the top-level binding (trampoline.c).
     * But here, we have to do it ourselves. */
    if (!data->pfn_dev_init) {
        *((const void **)cmd) = *(void **)device;
    } else {
        err = data->pfn_dev_init(device, (void *)cmd);
        assert(!err);
    }

    VkCommandBufferBeginInfo cbbi;
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.pNext = nullptr;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    cbbi.pInheritanceInfo = nullptr;

    err = pTable->BeginCommandBuffer(cmd, &cbbi);
    assert(!err);

    VkImageMemoryBarrier imb;
    imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imb.pNext = nullptr;
    imb.dstAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    imb.srcAccessMask = 0;
    imb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imb.image = data->fontGlyphsImage;
    imb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imb.subresourceRange.baseMipLevel = 0;
    imb.subresourceRange.levelCount = 1;
    imb.subresourceRange.baseArrayLayer = 0;
    imb.subresourceRange.layerCount = 1;
    imb.srcQueueFamilyIndex = data->graphicsQueueFamilyIndex;
    imb.dstQueueFamilyIndex = data->graphicsQueueFamilyIndex;

    pTable->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0 /* dependency flags */,
                               0, nullptr, /* memory barriers */
                               0, nullptr, /* buffer memory barriers */
                               1, &imb);   /* image memory barriers */

    pTable->EndCommandBuffer(cmd);
    data->fontUploadCmdBuffer = cmd;
    data->fontUploadComplete = false; /* we will schedule this at first present on this device */

#ifdef OVERLAY_DEBUG
    printf("Font upload done.\n");
#endif

    /* create a sampler to use with the texture */
    VkSamplerCreateInfo sci;
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.pNext = nullptr;
    sci.flags = 0;
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.mipLodBias = 0.0f;
    sci.anisotropyEnable = false;
    sci.maxAnisotropy = 1;
    sci.compareEnable = false;
    sci.compareOp = VK_COMPARE_OP_NEVER;
    sci.minLod = 0.0f;
    sci.maxLod = 0.0f;
    sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sci.unnormalizedCoordinates = VK_FALSE;

    err = pTable->CreateSampler(device, &sci, nullptr, &data->sampler);
    assert(!err);

    /* descriptor set stuff so we can use the texture from a shader. */
    VkDescriptorSetLayoutBinding dslb[1];
    dslb[0].binding = 0;
    dslb[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    dslb[0].descriptorCount = 1;
    dslb[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    dslb[0].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo dslci;
    memset(&dslci, 0, sizeof(dslci));
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.pNext = nullptr;
    dslci.bindingCount = 1;
    dslci.pBindings = dslb;

    err = pTable->CreateDescriptorSetLayout(device, &dslci, nullptr, &data->dsl);
    assert(!err);

    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &data->dsl;

    err = pTable->CreatePipelineLayout(device, &plci, nullptr, &data->pl);
    assert(!err);

    VkDescriptorPoolSize dtc[1];
    dtc[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    dtc[0].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci;
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.pNext = nullptr;
    dpci.flags = 0;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = dtc;

    err = pTable->CreateDescriptorPool(device, &dpci, nullptr, &data->desc_pool);
    assert(!err);

    VkDescriptorSetAllocateInfo dsai;
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.pNext = nullptr;
    dsai.descriptorPool = data->desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &data->dsl;
    err = pTable->AllocateDescriptorSets(device, &dsai, &data->desc_set);
    assert(!err);

    VkDescriptorImageInfo descs[1];
    descs[0].sampler = data->sampler;
    descs[0].imageView = data->fontGlyphsImageView;
    descs[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;  // TODO: cube does this, is it correct?

    VkWriteDescriptorSet writes[1];
    memset(&writes, 0, sizeof(writes));
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = data->desc_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = descs;

    pTable->UpdateDescriptorSets(device, 1, writes, 0, nullptr);

    VkFenceCreateInfo fci;
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.pNext = NULL;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    pTable->CreateFence(device, &fci, NULL, &data->fence);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice gpu, const VkDeviceCreateInfo *pCreateInfo,
                                                              const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
    VkLayerDeviceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);

    assert(chain_info->u.pLayerInfo);
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    PFN_vkCreateDevice fpCreateDevice = (PFN_vkCreateDevice)fpGetInstanceProcAddr(NULL, "vkCreateDevice");
    if (fpCreateDevice == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = fpCreateDevice(gpu, pCreateInfo, pAllocator, pDevice);
    if (result != VK_SUCCESS) {
        return result;
    }

    layer_data *my_device_data = GetLayerDataPtr(get_dispatch_key(*pDevice), layer_data_map);

    // Setup device dispatch table
    my_device_data->device_dispatch_table = new VkLayerDispatchTable;
    layer_init_device_dispatch_table(*pDevice, my_device_data->device_dispatch_table, fpGetDeviceProcAddr);

    // store the loader callback for initializing created dispatchable objects
    chain_info = get_chain_info(pCreateInfo, VK_LOADER_DATA_CALLBACK);
    if (chain_info) {
        my_device_data->pfn_dev_init = chain_info->u.pfnSetDeviceLoaderData;
    } else {
        my_device_data->pfn_dev_init = NULL;
    }

    uint32_t queue_family_count;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(gpu), layer_data_map);
    my_data->instance_dispatch_table->GetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, NULL);
    VkQueueFamilyProperties *queue_props = (VkQueueFamilyProperties *)malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
    if (queue_props == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    my_data->instance_dispatch_table->GetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, queue_props);
    my_device_data->graphicsQueueFamilyIndex = 0;
    for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
        if (queue_props[pCreateInfo->pQueueCreateInfos[i].queueFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            my_device_data->graphicsQueueFamilyIndex = pCreateInfo->pQueueCreateInfos[i].queueFamilyIndex;
            break;
        }
    }
    free(queue_props);

    after_device_create(gpu, *pDevice, my_device_data);
    return result;
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    my_data->Cleanup();
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    pTable->DeviceWaitIdle(device);
    pTable->DestroyDevice(device, pAllocator);
    delete pTable;
    layer_data_map.erase(key);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                                                                const VkAllocationCallbacks *pAllocator, VkInstance *pInstance) {
    VkLayerInstanceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);

    assert(chain_info->u.pLayerInfo);
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateInstance fpCreateInstance = (PFN_vkCreateInstance)fpGetInstanceProcAddr(NULL, "vkCreateInstance");
    if (fpCreateInstance == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = fpCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS) return result;

    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(*pInstance), layer_data_map);
    my_data->instance_dispatch_table = new VkLayerInstanceDispatchTable;
    layer_init_instance_dispatch_table(*pInstance, my_data->instance_dispatch_table, fpGetInstanceProcAddr);

    init_overlay(my_data);
    return result;
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    pTable->DestroyInstance(instance, pAllocator);
    delete pTable;
    layer_data_map.erase(key);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR *pCreateInfo,
                                                                    const VkAllocationCallbacks *pAllocator,
                                                                    VkSwapchainKHR *pSwapChain) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result = my_data->pfnCreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapChain);

    if (result == VK_SUCCESS) {
        auto &data = (*my_data->swapChains)[*pSwapChain];
        data = new SwapChainData;
        data->width = pCreateInfo->imageExtent.width;
        data->height = pCreateInfo->imageExtent.height;
        data->format = pCreateInfo->imageFormat;

#ifdef OVERLAY_DEBUG
        printf(
            "Creating resources for scribbling on swapchain format %u width "
            "%u height %u\n",
            data->format, data->width, data->height);
#endif

        /* Create a renderpass for drawing into this swapchain */
        VkAttachmentDescription ad;
        ad.flags = 0;
        ad.format = data->format;
        ad.samples = VK_SAMPLE_COUNT_1_BIT;
        ad.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ad.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ad.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ad.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        /* TODO: deal with the image possibly being in a different
         * layout - we need to care about general,
         * VK_IMAGE_LAYOUT_PRESENT_SOURCE_WSI,
         * etc etc
         */
        ad.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ad.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference ar;
        ar.attachment = 0;
        /* TODO: see previous layout comment */
        ar.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference dr;
        dr.attachment = VK_ATTACHMENT_UNUSED;
        dr.layout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkSubpassDescription sd;
        sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sd.flags = 0;
        sd.inputAttachmentCount = 0;
        sd.pInputAttachments = nullptr;
        sd.colorAttachmentCount = 1;
        sd.pColorAttachments = &ar;
        sd.pResolveAttachments = nullptr;
        sd.pDepthStencilAttachment = &dr;
        /* TODO: do we need to mark the color attachment here? */
        sd.preserveAttachmentCount = 0;
        sd.pPreserveAttachments = nullptr;

        VkRenderPassCreateInfo rpci;
        rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpci.pNext = nullptr;
        rpci.flags = 0;
        rpci.attachmentCount = 1;
        rpci.pAttachments = &ad;
        rpci.subpassCount = 1;
        rpci.pSubpasses = &sd;
        rpci.dependencyCount = 0;
        rpci.pDependencies = nullptr;

        pTable->CreateRenderPass(device, &rpci, nullptr, &data->render_pass);

        /* Create the pipeline to use in this renderpass */
        VkPipelineShaderStageCreateInfo stages[2];
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].pNext = nullptr;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = my_data->vsShaderModule;
        stages[0].pName = "main";
        stages[0].flags = 0;
        stages[0].pSpecializationInfo = nullptr;

        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].pNext = nullptr;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = my_data->fsShaderModule;
        stages[1].pName = "main";
        stages[1].flags = 0;
        stages[1].pSpecializationInfo = nullptr;

        VkPipelineInputAssemblyStateCreateInfo piasci;
        memset(&piasci, 0, sizeof(piasci));
        piasci.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        piasci.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport;
        memset(&viewport, 0, sizeof(viewport));
        viewport.width = (float)data->width;
        viewport.height = (float)data->height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor;
        memset(&scissor, 0, sizeof(scissor));
        scissor.extent.width = data->width;
        scissor.extent.height = data->height;

        VkPipelineViewportStateCreateInfo pvsci;
        memset(&pvsci, 0, sizeof(pvsci));
        pvsci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        pvsci.viewportCount = 1;
        pvsci.pViewports = &viewport;
        pvsci.scissorCount = 1;
        pvsci.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo prsci;
        memset(&prsci, 0, sizeof(prsci));
        prsci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        prsci.polygonMode = VK_POLYGON_MODE_FILL;
        prsci.cullMode = VK_CULL_MODE_NONE;
        prsci.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo pmsci;
        memset(&pmsci, 0, sizeof(pmsci));
        pmsci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        pmsci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo pdssci;
        memset(&pdssci, 0, sizeof(pdssci));
        pdssci.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        pdssci.minDepthBounds = 0.0f;
        pdssci.maxDepthBounds = 1.0f;

        VkPipelineColorBlendAttachmentState pcbas;
        memset(&pcbas, 0, sizeof(pcbas));
        pcbas.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo pcbsci;
        memset(&pcbsci, 0, sizeof(pcbsci));
        pcbsci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        pcbsci.attachmentCount = 1;
        pcbsci.pAttachments = &pcbas;
        pcbsci.blendConstants[0] = 1.0f;
        pcbsci.blendConstants[1] = 1.0f;
        pcbsci.blendConstants[2] = 1.0f;
        pcbsci.blendConstants[3] = 1.0f;

        VkVertexInputBindingDescription bindings[] = {
            {0, sizeof(vertex), VK_VERTEX_INPUT_RATE_VERTEX},
        };

        VkVertexInputAttributeDescription attribs[] = {
            {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(vertex, x)}, {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(vertex, u)},
        };

        VkPipelineVertexInputStateCreateInfo pvisci;
        memset(&pvisci, 0, sizeof(pvisci));
        pvisci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        pvisci.vertexBindingDescriptionCount = sizeof(bindings) / sizeof(*bindings);
        pvisci.pVertexBindingDescriptions = &bindings[0];
        pvisci.vertexAttributeDescriptionCount = sizeof(attribs) / sizeof(*attribs);
        pvisci.pVertexAttributeDescriptions = &attribs[0];

        VkGraphicsPipelineCreateInfo gpci;
        gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.pNext = nullptr;
        gpci.stageCount = sizeof(stages) / sizeof(*stages);
        gpci.pStages = &stages[0];

        gpci.pVertexInputState = &pvisci;
        gpci.pInputAssemblyState = &piasci;
        gpci.pTessellationState = nullptr;
        gpci.pViewportState = &pvsci;
        gpci.pRasterizationState = &prsci;
        gpci.pMultisampleState = &pmsci;
        gpci.pDepthStencilState = &pdssci;
        gpci.pColorBlendState = &pcbsci;
        gpci.pDynamicState = nullptr;
        gpci.flags = 0;
        gpci.layout = my_data->pl;
        gpci.renderPass = data->render_pass;
        gpci.subpass = 0;
        gpci.basePipelineHandle = VK_NULL_HANDLE;
        gpci.basePipelineIndex = 0;

        pTable->CreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &data->pipeline);
    }

    return result;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapChain, uint32_t *pCount,
                                                                       VkImage *pImages) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result = my_data->pfnGetSwapchainImagesKHR(device, swapChain, pCount, pImages);
    VkResult U_ASSERT_ONLY err;

    /* GetSwapChainImagesWSI may be called without an images buffer, in which
     * case it
     * just returns the count to the caller. We're only interested in acting on
     * the
     * /actual/ fetch of the images.
     */
    if (pImages) {
        auto data = (*my_data->swapChains)[swapChain];

        for (uint32_t i = 0; i < *pCount; i++) {
            /* Create attachment view for each */
            VkImageViewCreateInfo ivci;
            ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ivci.pNext = nullptr;
            ivci.format = data->format;
            ivci.components.r = VK_COMPONENT_SWIZZLE_R;
            ivci.components.g = VK_COMPONENT_SWIZZLE_G;
            ivci.components.b = VK_COMPONENT_SWIZZLE_B;
            ivci.components.a = VK_COMPONENT_SWIZZLE_A;
            ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ivci.subresourceRange.baseMipLevel = 0;
            ivci.subresourceRange.levelCount = 1;
            ivci.subresourceRange.baseArrayLayer = 0;
            ivci.subresourceRange.layerCount = 1;
            ivci.image = pImages[i];
            ivci.flags = 0;

            VkImageView v;
            pTable->CreateImageView(device, &ivci, nullptr, &v);

            /* Create framebuffer for each */
            VkFramebufferCreateInfo fci;
            fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fci.pNext = nullptr;
            fci.flags = 0;
            fci.renderPass = data->render_pass;
            fci.attachmentCount = 1;
            fci.pAttachments = &v;
            fci.width = data->width;
            fci.height = data->height;
            fci.layers = 1;

            VkFramebuffer fb;
            pTable->CreateFramebuffer(device, &fci, nullptr, &fb);

            /* Create command buffer for each */
            VkCommandBufferAllocateInfo cbai;
            cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cbai.pNext = nullptr;
            cbai.commandPool = my_data->pool;
            cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 1;

            VkCommandBuffer cmd;
            pTable->AllocateCommandBuffers(device, &cbai, &cmd);

            /* We have just created a dispatchable object, but the dispatch
             * table has not been placed in the object yet.
             * When a "normal" application creates a command buffer,
             * the dispatch table is installed by the top-level binding
             * (trampoline.c).
             * But here, we have to do it ourselves. */

            if (!my_data->pfn_dev_init) {
                *((const void **)cmd) = *(void **)device;
            } else {
                err = my_data->pfn_dev_init(device, (void *)cmd);
                assert(!err);
            }

            /* Create vertex buffer */
            VkBufferCreateInfo bci;
            memset(&bci, 0, sizeof(bci));
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            bci.size = sizeof(vertex) * MAX_TEXT_VERTICES;

            VkBuffer buf;
            err = pTable->CreateBuffer(device, &bci, nullptr, &buf);
            assert(!err);

            VkMemoryRequirements mem_reqs;
            pTable->GetBufferMemoryRequirements(device, buf, &mem_reqs);
            assert(!err);

            VkMemoryAllocateInfo mem_alloc;
            memset(&mem_alloc, 0, sizeof(mem_alloc));
            mem_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mem_alloc.allocationSize = mem_reqs.size;
            mem_alloc.memoryTypeIndex = choose_memory_type(
                my_data->gpu, mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            VkDeviceMemory mem;
            err = pTable->AllocateMemory(device, &mem_alloc, nullptr, &mem);
            assert(!err);

            err = pTable->BindBufferMemory(device, buf, mem, 0);
            assert(!err);

            auto imageData = new WsiImageData;
            imageData->image = pImages[i];
            imageData->view = v;
            imageData->framebuffer = fb;
            imageData->cmd = cmd;
            imageData->vertexBuffer = buf;
            imageData->vertexBufferMemory = mem;
            imageData->numVertices = 0;
            imageData->vertexBufferSize = mem_alloc.allocationSize;

            data->presentableImages.push_back(imageData);
        }
    }
    return result;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits,
                                                             VkFence fence) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(queue), layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;

    my_data->cmdBuffersThisFrame += submitCount;  // XXX WRONG

    return pTable->QueueSubmit(queue, submitCount, pSubmits, fence);
}

static void before_present(VkQueue queue, layer_data *my_data, SwapChainData *swapChain, unsigned imageIndex) {
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;

    if (!my_data->fontUploadComplete) {
        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.pNext = nullptr;
        si.waitSemaphoreCount = 0;
        si.commandBufferCount = 1;
        si.signalSemaphoreCount = 0;
        si.pCommandBuffers = &my_data->fontUploadCmdBuffer;

        pTable->QueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        my_data->fontUploadComplete = true;
#ifdef OVERLAY_DEBUG
        printf("Font image layout transition queued\n");
#endif
    }

    WsiImageData *id = swapChain->presentableImages[imageIndex];

    /* update the overlay content */

    vertex *vertices = nullptr;

    /* guaranteed not in flight due to WSI surface being available */
    VkResult U_ASSERT_ONLY err =
        pTable->MapMemory(my_data->dev, id->vertexBufferMemory, 0, id->vertexBufferSize, 0, (void **)&vertices);
    assert(!err);

    /* write vertices for string in here */
    id->numVertices = fill_vertex_buffer(my_data, vertices, imageIndex);

    pTable->UnmapMemory(my_data->dev, id->vertexBufferMemory);

    /* JIT record a command buffer to draw the overlay */

    VkCommandBufferBeginInfo cbbi;
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.pNext = nullptr;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    cbbi.pInheritanceInfo = nullptr;

    VkImageMemoryBarrier imb;
    imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imb.pNext = nullptr;
    imb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    imb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imb.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    imb.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    imb.image = id->image;
    imb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imb.subresourceRange.baseMipLevel = 0;
    imb.subresourceRange.levelCount = 1;
    imb.subresourceRange.baseArrayLayer = 0;
    imb.subresourceRange.layerCount = 1;
    imb.srcQueueFamilyIndex = my_data->graphicsQueueFamilyIndex;
    imb.dstQueueFamilyIndex = my_data->graphicsQueueFamilyIndex;

    VkRenderPassBeginInfo rpbi;
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.pNext = nullptr;
    rpbi.renderPass = swapChain->render_pass;
    rpbi.framebuffer = id->framebuffer;
    rpbi.renderArea.offset.x = 0;
    rpbi.renderArea.offset.y = 0;
    rpbi.renderArea.extent.width = swapChain->width;
    rpbi.renderArea.extent.height = swapChain->height;
    rpbi.clearValueCount = 0;
    rpbi.pClearValues = nullptr;

    pTable->BeginCommandBuffer(id->cmd, &cbbi);
    pTable->CmdPipelineBarrier(id->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               0 /* dependency flags */, 0, nullptr, /* memory barriers */
                               0, nullptr,                           /* buffer memory barriers */
                               1, &imb);                             /* image memory barriers */
    pTable->CmdBeginRenderPass(id->cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    pTable->CmdBindPipeline(id->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, swapChain->pipeline);
    pTable->CmdBindDescriptorSets(id->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, my_data->pl, 0, 1, &my_data->desc_set, 0, nullptr);

    VkDeviceSize offsets[] = {0};
    VkBuffer buffers[] = {id->vertexBuffer};

    pTable->CmdBindVertexBuffers(id->cmd, 0, 1, buffers, offsets);

    pTable->CmdDraw(id->cmd, id->numVertices, 1, 0, 0);

    pTable->CmdEndRenderPass(id->cmd);
    pTable->EndCommandBuffer(id->cmd);

    /* Schedule this command buffer for execution. TODO: Do we need to protect
     * ourselves from an app that didn't wait for the presentation image to
     * be idle before mangling it? If the app is well-behaved, our command
     * buffer is guaranteed to have been retired before the app tries to
     * present it again.
     */
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = nullptr;
    si.waitSemaphoreCount = 0;
    si.commandBufferCount = 1;
    si.signalSemaphoreCount = 0;
    si.pCommandBuffers = &id->cmd;
    pTable->WaitForFences(my_data->dev, 1, &my_data->fence, VK_TRUE, UINT64_MAX);
    pTable->ResetFences(my_data->dev, 1, &my_data->fence);
    pTable->QueueSubmit(queue, 1, &si, my_data->fence);

    /* Reset per-frame stats */
    my_data->cmdBuffersThisFrame = 0;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pPresentInfo) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(queue), layer_data_map);

    for (uint32_t i = 0; i < pPresentInfo->swapchainCount; i++) {
        auto data = my_data->swapChains->find(pPresentInfo->pSwapchains[i]);
        assert(data != my_data->swapChains->end());

        before_present(queue, my_data, data->second, pPresentInfo->pImageIndices[i]);
    }

    VkResult result = my_data->pfnQueuePresentKHR(queue, pPresentInfo);
    return result;
}

void WsiImageData::Cleanup(VkDevice dev) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(dev), layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    pTable->DeviceWaitIdle(dev);

    // XXX: needs device data
    //    pTable->FreeCommandBuffers(dev, cmd, nullptr);
    pTable->DestroyFramebuffer(dev, framebuffer, nullptr);
    pTable->DestroyImageView(dev, view, nullptr);
    pTable->DestroyBuffer(dev, vertexBuffer, nullptr);
    pTable->FreeMemory(dev, vertexBufferMemory, nullptr);
}

void SwapChainData::Cleanup(VkDevice dev) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(dev), layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;

    for (uint32_t i = 0; i < presentableImages.size(); i++) {
        presentableImages[i]->Cleanup(dev);
        delete presentableImages[i];
    }

    presentableImages.clear();

    pTable->DestroyPipeline(dev, pipeline, nullptr);
    pTable->DestroyRenderPass(dev, render_pass, nullptr);
}

void layer_data::Cleanup() {
    VkLayerDispatchTable *pTable = this->device_dispatch_table;

    pTable->DestroySampler(dev, sampler, nullptr);
    pTable->DestroyDescriptorPool(dev, desc_pool, nullptr);
    pTable->DestroyPipelineLayout(dev, pl, nullptr);
    pTable->DestroyDescriptorSetLayout(dev, dsl, nullptr);
    pTable->DestroyImageView(dev, fontGlyphsImageView, nullptr);
    pTable->DestroyImage(dev, fontGlyphsImage, nullptr);
    pTable->FreeMemory(dev, fontGlyphsMemory, nullptr);

    pTable->FreeCommandBuffers(dev, pool, 1, &fontUploadCmdBuffer);
    pTable->DestroyCommandPool(dev, pool, nullptr);

    pTable->DestroyShaderModule(dev, vsShaderModule, nullptr);
    pTable->DestroyShaderModule(dev, fsShaderModule, nullptr);
    pTable->DestroyFence(dev, fence, nullptr);
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain,
                                                                 const VkAllocationCallbacks *pAllocator) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);

    /* Clean up our resources associated with this swapchain */
    auto it = my_data->swapChains->find(swapchain);
    assert(it != my_data->swapChains->end());

    it->second->Cleanup(device);
    delete it->second;
    my_data->swapChains->erase(it->first);

    my_data->pfnDestroySwapchainKHR(device, swapchain, pAllocator);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice dev, const char *funcName) {
#define ADD_HOOK(fn) \
    if (!strncmp(#fn, funcName, sizeof(#fn))) return (PFN_vkVoidFunction)fn

    ADD_HOOK(vkGetDeviceProcAddr);
    ADD_HOOK(vkDestroyDevice);
    ADD_HOOK(vkCreateSwapchainKHR);
    ADD_HOOK(vkGetSwapchainImagesKHR);
    ADD_HOOK(vkQueuePresentKHR);
    ADD_HOOK(vkDestroySwapchainKHR);
    ADD_HOOK(vkQueueSubmit);
#undef ADD_HOOK

    if (dev == NULL) return NULL;

    layer_data *dev_data;
    dev_data = GetLayerDataPtr(get_dispatch_key(dev), layer_data_map);
    VkLayerDispatchTable *pTable = dev_data->device_dispatch_table;

    if (pTable->GetDeviceProcAddr == NULL) return NULL;
    return pTable->GetDeviceProcAddr(dev, funcName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *funcName) {
#define ADD_HOOK(fn) \
    if (!strncmp(#fn, funcName, sizeof(#fn))) return (PFN_vkVoidFunction)fn

    ADD_HOOK(vkCreateInstance);
    ADD_HOOK(vkCreateDevice);
    ADD_HOOK(vkDestroyInstance);
#undef ADD_HOOK

    if (instance == NULL) return NULL;

    layer_data *instance_data;
    instance_data = GetLayerDataPtr(get_dispatch_key(instance), layer_data_map);
    VkLayerInstanceDispatchTable *pTable = instance_data->instance_dispatch_table;

    if (pTable->GetInstanceProcAddr == NULL) return NULL;
    return pTable->GetInstanceProcAddr(instance, funcName);
}
