/*
 * Copyright (c) 2015-2016 The Khronos Group Inc.
 * Copyright (c) 2015-2016 Valve Corporation
 * Copyright (c) 2015-2016 LunarG, Inc.
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
 * Author: Courtney Goeltzenleuchter <courtney@LunarG.com>
 * Author: David Pinedo <david@lunarg.com>
 * Author: Mark Lobodzinski <mark@lunarg.com>
 * Author: Rene Lindsay <rene@lunarg.com>
 * Author: Jeremy Kniager <jeremyk@lunarg.com>
 */

#ifdef __GNUC__
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#else
#define strndup(p, n) strdup(p)
#endif

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif  // _WIN32

#if defined(VK_USE_PLATFORM_XLIB_KHR) || defined(VK_USE_PLATFORM_XCB_KHR)
#include <X11/Xutil.h>
#endif

#if defined(VK_USE_PLATFORM_MIR_KHR)
#warning "Vulkaninfo does not have code for Mir at this time"
#endif

#include <vulkan/vulkan.h>

#define ERR(err) printf("%s:%d: failed with %s\n", __FILE__, __LINE__, VkResultString(err));

#ifdef _WIN32

#define snprintf _snprintf
#define strdup   _strdup

// Returns nonzero if the console is used only for this process. Will return
// zero if another process (such as cmd.exe) is also attached.
static int ConsoleIsExclusive(void) {
    DWORD pids[2];
    DWORD num_pids = GetConsoleProcessList(pids, ARRAYSIZE(pids));
    return num_pids <= 1;
}

#define WAIT_FOR_CONSOLE_DESTROY                   \
    do {                                           \
        if (ConsoleIsExclusive()) Sleep(INFINITE); \
    } while (0)
#else
#define WAIT_FOR_CONSOLE_DESTROY
#endif

#define ERR_EXIT(err)             \
    do {                          \
        ERR(err);                 \
        fflush(stdout);           \
        WAIT_FOR_CONSOLE_DESTROY; \
        exit(-1);                 \
    } while (0)

#if defined(NDEBUG) && defined(__GNUC__)
#define U_ASSERT_ONLY __attribute__((unused))
#else
#define U_ASSERT_ONLY
#endif

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))

#define MAX_QUEUE_TYPES 5
#define APP_SHORT_NAME "vulkaninfo"

struct VkStructureHeader {
    VkStructureType sType;
    void *pNext;
};

struct AppGpu;

struct AppDev {
    struct AppGpu *gpu; /* point back to the GPU */

    VkDevice obj;

    VkFormatProperties format_props[VK_FORMAT_RANGE_SIZE];
    VkFormatProperties2KHR format_props2[VK_FORMAT_RANGE_SIZE];
};

struct LayerExtensionList {
    VkLayerProperties layer_properties;
    uint32_t extension_count;
    VkExtensionProperties *extension_properties;
};

struct AppInstance {
    VkInstance instance;
    uint32_t global_layer_count;
    struct LayerExtensionList *global_layers;
    uint32_t global_extension_count;
    VkExtensionProperties *global_extensions;  // Instance Extensions

    const char **inst_extensions;
    uint32_t inst_extensions_count;

    PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR;
    PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR;
    PFN_vkGetPhysicalDeviceFormatProperties2KHR vkGetPhysicalDeviceFormatProperties2KHR;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR vkGetPhysicalDeviceQueueFamilyProperties2KHR;
    PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR;
    PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR;
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceCapabilities2KHR;
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT vkGetPhysicalDeviceSurfaceCapabilities2EXT;

    VkSurfaceCapabilitiesKHR surface_capabilities;
    VkSurfaceCapabilities2KHR surface_capabilities2;
    VkSharedPresentSurfaceCapabilitiesKHR shared_surface_capabilities;
    VkSurfaceCapabilities2EXT surface_capabilities2_ext;

    VkSurfaceKHR surface;
    int width, height;

#ifdef VK_USE_PLATFORM_WIN32_KHR
    HINSTANCE h_instance;  // Windows Instance
    HWND h_wnd;            // window handle
#elif VK_USE_PLATFORM_XCB_KHR
    xcb_connection_t *xcb_connection;
    xcb_screen_t *xcb_screen;
    xcb_window_t xcb_window;
#elif VK_USE_PLATFORM_XLIB_KHR
    Display *xlib_display;
    Window xlib_window;
#elif VK_USE_PLATFORM_ANDROID_KHR  // TODO
    ANativeWindow *window;
#endif
};

struct AppGpu {
    uint32_t id;
    VkPhysicalDevice obj;

    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceProperties2KHR props2;

    uint32_t queue_count;
    VkQueueFamilyProperties *queue_props;
    VkQueueFamilyProperties2KHR *queue_props2;
    VkDeviceQueueCreateInfo *queue_reqs;

    struct AppInstance *inst;

    VkPhysicalDeviceMemoryProperties memory_props;
    VkPhysicalDeviceMemoryProperties2KHR memory_props2;

    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceFeatures2KHR features2;
    VkPhysicalDevice limits;

    uint32_t device_extension_count;
    VkExtensionProperties *device_extensions;

    struct AppDev dev;
};

static VKAPI_ATTR VkBool32 VKAPI_CALL DbgCallback(VkFlags msgFlags, VkDebugReportObjectTypeEXT objType, uint64_t srcObject,
                                                  size_t location, int32_t msgCode, const char *pLayerPrefix, const char *pMsg,
                                                  void *pUserData) {
    char *message = (char *)malloc(strlen(pMsg) + 100);

    assert(message);

    if (msgFlags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        sprintf(message, "ERROR: [%s] Code %d : %s", pLayerPrefix, msgCode, pMsg);
    } else if (msgFlags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
        sprintf(message, "WARNING: [%s] Code %d : %s", pLayerPrefix, msgCode, pMsg);
    } else if (msgFlags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
        sprintf(message, "INFO: [%s] Code %d : %s", pLayerPrefix, msgCode, pMsg);
    } else if (msgFlags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) {
        sprintf(message, "DEBUG: [%s] Code %d : %s", pLayerPrefix, msgCode, pMsg);
    }

    printf("%s\n", message);
    fflush(stdout);
    free(message);

    /*
     * false indicates that layer should not bail-out of an
     * API call that had validation failures. This may mean that the
     * app dies inside the driver due to invalid parameter(s).
     * That's what would happen without validation layers, so we'll
     * keep that behavior here.
     */
    return false;
}

static const char *VkResultString(VkResult err) {
    switch (err) {
#define STR(r) \
    case r:    \
        return #r
        STR(VK_SUCCESS);
        STR(VK_NOT_READY);
        STR(VK_TIMEOUT);
        STR(VK_EVENT_SET);
        STR(VK_EVENT_RESET);
        STR(VK_ERROR_INITIALIZATION_FAILED);
        STR(VK_ERROR_OUT_OF_HOST_MEMORY);
        STR(VK_ERROR_OUT_OF_DEVICE_MEMORY);
        STR(VK_ERROR_DEVICE_LOST);
        STR(VK_ERROR_LAYER_NOT_PRESENT);
        STR(VK_ERROR_EXTENSION_NOT_PRESENT);
        STR(VK_ERROR_MEMORY_MAP_FAILED);
        STR(VK_ERROR_INCOMPATIBLE_DRIVER);
#undef STR
        default:
            return "UNKNOWN_RESULT";
    }
}

static const char *VkPhysicalDeviceTypeString(VkPhysicalDeviceType type) {
    switch (type) {
#define STR(r)                        \
    case VK_PHYSICAL_DEVICE_TYPE_##r: \
        return #r
        STR(OTHER);
        STR(INTEGRATED_GPU);
        STR(DISCRETE_GPU);
        STR(VIRTUAL_GPU);
        STR(CPU);
#undef STR
        default:
            return "UNKNOWN_DEVICE";
    }
}

static const char *VkFormatString(VkFormat fmt) {
    switch (fmt) {
#define STR(r)          \
    case VK_FORMAT_##r: \
        return #r
        STR(UNDEFINED);
        STR(R4G4_UNORM_PACK8);
        STR(R4G4B4A4_UNORM_PACK16);
        STR(B4G4R4A4_UNORM_PACK16);
        STR(R5G6B5_UNORM_PACK16);
        STR(B5G6R5_UNORM_PACK16);
        STR(R5G5B5A1_UNORM_PACK16);
        STR(B5G5R5A1_UNORM_PACK16);
        STR(A1R5G5B5_UNORM_PACK16);
        STR(R8_UNORM);
        STR(R8_SNORM);
        STR(R8_USCALED);
        STR(R8_SSCALED);
        STR(R8_UINT);
        STR(R8_SINT);
        STR(R8_SRGB);
        STR(R8G8_UNORM);
        STR(R8G8_SNORM);
        STR(R8G8_USCALED);
        STR(R8G8_SSCALED);
        STR(R8G8_UINT);
        STR(R8G8_SINT);
        STR(R8G8_SRGB);
        STR(R8G8B8_UNORM);
        STR(R8G8B8_SNORM);
        STR(R8G8B8_USCALED);
        STR(R8G8B8_SSCALED);
        STR(R8G8B8_UINT);
        STR(R8G8B8_SINT);
        STR(R8G8B8_SRGB);
        STR(B8G8R8_UNORM);
        STR(B8G8R8_SNORM);
        STR(B8G8R8_USCALED);
        STR(B8G8R8_SSCALED);
        STR(B8G8R8_UINT);
        STR(B8G8R8_SINT);
        STR(B8G8R8_SRGB);
        STR(R8G8B8A8_UNORM);
        STR(R8G8B8A8_SNORM);
        STR(R8G8B8A8_USCALED);
        STR(R8G8B8A8_SSCALED);
        STR(R8G8B8A8_UINT);
        STR(R8G8B8A8_SINT);
        STR(R8G8B8A8_SRGB);
        STR(B8G8R8A8_UNORM);
        STR(B8G8R8A8_SNORM);
        STR(B8G8R8A8_USCALED);
        STR(B8G8R8A8_SSCALED);
        STR(B8G8R8A8_UINT);
        STR(B8G8R8A8_SINT);
        STR(B8G8R8A8_SRGB);
        STR(A8B8G8R8_UNORM_PACK32);
        STR(A8B8G8R8_SNORM_PACK32);
        STR(A8B8G8R8_USCALED_PACK32);
        STR(A8B8G8R8_SSCALED_PACK32);
        STR(A8B8G8R8_UINT_PACK32);
        STR(A8B8G8R8_SINT_PACK32);
        STR(A8B8G8R8_SRGB_PACK32);
        STR(A2R10G10B10_UNORM_PACK32);
        STR(A2R10G10B10_SNORM_PACK32);
        STR(A2R10G10B10_USCALED_PACK32);
        STR(A2R10G10B10_SSCALED_PACK32);
        STR(A2R10G10B10_UINT_PACK32);
        STR(A2R10G10B10_SINT_PACK32);
        STR(A2B10G10R10_UNORM_PACK32);
        STR(A2B10G10R10_SNORM_PACK32);
        STR(A2B10G10R10_USCALED_PACK32);
        STR(A2B10G10R10_SSCALED_PACK32);
        STR(A2B10G10R10_UINT_PACK32);
        STR(A2B10G10R10_SINT_PACK32);
        STR(R16_UNORM);
        STR(R16_SNORM);
        STR(R16_USCALED);
        STR(R16_SSCALED);
        STR(R16_UINT);
        STR(R16_SINT);
        STR(R16_SFLOAT);
        STR(R16G16_UNORM);
        STR(R16G16_SNORM);
        STR(R16G16_USCALED);
        STR(R16G16_SSCALED);
        STR(R16G16_UINT);
        STR(R16G16_SINT);
        STR(R16G16_SFLOAT);
        STR(R16G16B16_UNORM);
        STR(R16G16B16_SNORM);
        STR(R16G16B16_USCALED);
        STR(R16G16B16_SSCALED);
        STR(R16G16B16_UINT);
        STR(R16G16B16_SINT);
        STR(R16G16B16_SFLOAT);
        STR(R16G16B16A16_UNORM);
        STR(R16G16B16A16_SNORM);
        STR(R16G16B16A16_USCALED);
        STR(R16G16B16A16_SSCALED);
        STR(R16G16B16A16_UINT);
        STR(R16G16B16A16_SINT);
        STR(R16G16B16A16_SFLOAT);
        STR(R32_UINT);
        STR(R32_SINT);
        STR(R32_SFLOAT);
        STR(R32G32_UINT);
        STR(R32G32_SINT);
        STR(R32G32_SFLOAT);
        STR(R32G32B32_UINT);
        STR(R32G32B32_SINT);
        STR(R32G32B32_SFLOAT);
        STR(R32G32B32A32_UINT);
        STR(R32G32B32A32_SINT);
        STR(R32G32B32A32_SFLOAT);
        STR(R64_UINT);
        STR(R64_SINT);
        STR(R64_SFLOAT);
        STR(R64G64_UINT);
        STR(R64G64_SINT);
        STR(R64G64_SFLOAT);
        STR(R64G64B64_UINT);
        STR(R64G64B64_SINT);
        STR(R64G64B64_SFLOAT);
        STR(R64G64B64A64_UINT);
        STR(R64G64B64A64_SINT);
        STR(R64G64B64A64_SFLOAT);
        STR(B10G11R11_UFLOAT_PACK32);
        STR(E5B9G9R9_UFLOAT_PACK32);
        STR(D16_UNORM);
        STR(X8_D24_UNORM_PACK32);
        STR(D32_SFLOAT);
        STR(S8_UINT);
        STR(D16_UNORM_S8_UINT);
        STR(D24_UNORM_S8_UINT);
        STR(D32_SFLOAT_S8_UINT);
        STR(BC1_RGB_UNORM_BLOCK);
        STR(BC1_RGB_SRGB_BLOCK);
        STR(BC2_UNORM_BLOCK);
        STR(BC2_SRGB_BLOCK);
        STR(BC3_UNORM_BLOCK);
        STR(BC3_SRGB_BLOCK);
        STR(BC4_UNORM_BLOCK);
        STR(BC4_SNORM_BLOCK);
        STR(BC5_UNORM_BLOCK);
        STR(BC5_SNORM_BLOCK);
        STR(BC6H_UFLOAT_BLOCK);
        STR(BC6H_SFLOAT_BLOCK);
        STR(BC7_UNORM_BLOCK);
        STR(BC7_SRGB_BLOCK);
        STR(ETC2_R8G8B8_UNORM_BLOCK);
        STR(ETC2_R8G8B8A1_UNORM_BLOCK);
        STR(ETC2_R8G8B8A8_UNORM_BLOCK);
        STR(EAC_R11_UNORM_BLOCK);
        STR(EAC_R11_SNORM_BLOCK);
        STR(EAC_R11G11_UNORM_BLOCK);
        STR(EAC_R11G11_SNORM_BLOCK);
        STR(ASTC_4x4_UNORM_BLOCK);
        STR(ASTC_4x4_SRGB_BLOCK);
        STR(ASTC_5x4_UNORM_BLOCK);
        STR(ASTC_5x4_SRGB_BLOCK);
        STR(ASTC_5x5_UNORM_BLOCK);
        STR(ASTC_5x5_SRGB_BLOCK);
        STR(ASTC_6x5_UNORM_BLOCK);
        STR(ASTC_6x5_SRGB_BLOCK);
        STR(ASTC_6x6_UNORM_BLOCK);
        STR(ASTC_6x6_SRGB_BLOCK);
        STR(ASTC_8x5_UNORM_BLOCK);
        STR(ASTC_8x5_SRGB_BLOCK);
        STR(ASTC_8x6_UNORM_BLOCK);
        STR(ASTC_8x6_SRGB_BLOCK);
        STR(ASTC_8x8_UNORM_BLOCK);
        STR(ASTC_8x8_SRGB_BLOCK);
        STR(ASTC_10x5_UNORM_BLOCK);
        STR(ASTC_10x5_SRGB_BLOCK);
        STR(ASTC_10x6_UNORM_BLOCK);
        STR(ASTC_10x6_SRGB_BLOCK);
        STR(ASTC_10x8_UNORM_BLOCK);
        STR(ASTC_10x8_SRGB_BLOCK);
        STR(ASTC_10x10_UNORM_BLOCK);
        STR(ASTC_10x10_SRGB_BLOCK);
        STR(ASTC_12x10_UNORM_BLOCK);
        STR(ASTC_12x10_SRGB_BLOCK);
        STR(ASTC_12x12_UNORM_BLOCK);
        STR(ASTC_12x12_SRGB_BLOCK);
#undef STR
        default:
            return "UNKNOWN_FORMAT";
    }
}
#if defined(VK_USE_PLATFORM_XCB_KHR) || defined(VK_USE_PLATFORM_XLIB_KHR) || defined(VK_USE_PLATFORM_WIN32_KHR)
static const char *VkPresentModeString(VkPresentModeKHR mode) {
    switch (mode) {
#define STR(r)                \
    case VK_PRESENT_MODE_##r: \
        return #r
        STR(IMMEDIATE_KHR);
        STR(MAILBOX_KHR);
        STR(FIFO_KHR);
        STR(FIFO_RELAXED_KHR);
#undef STR
        default:
            return "UNKNOWN_FORMAT";
    }
}
#endif

static bool CheckExtensionEnabled(const char *extension_to_check, const char **extension_list, uint32_t extension_count) {
    for (uint32_t i = 0; i < extension_count; i++) {
        if (!strcmp(extension_to_check, extension_list[i])) return true;
    }
    return false;
}

static void AppDevInitFormats(struct AppDev *dev) {
    VkFormat f;
    for (f = 0; f < VK_FORMAT_RANGE_SIZE; f++) {
        const VkFormat fmt = f;
        vkGetPhysicalDeviceFormatProperties(dev->gpu->obj, fmt, &dev->format_props[f]);

        if (CheckExtensionEnabled(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, dev->gpu->inst->inst_extensions,
                                  dev->gpu->inst->inst_extensions_count)) {
            dev->format_props2[f].sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2_KHR;
            dev->format_props2[f].pNext = NULL;
            dev->gpu->inst->vkGetPhysicalDeviceFormatProperties2KHR(dev->gpu->obj, fmt, &dev->format_props2[f]);
        }
    }
}

static void ExtractVersion(uint32_t version, uint32_t *major, uint32_t *minor, uint32_t *patch) {
    *major = version >> 22;
    *minor = (version >> 12) & 0x3ff;
    *patch = version & 0xfff;
}

static void AppGetPhysicalDeviceLayerExtensions(struct AppGpu *gpu, char *layer_name, uint32_t *extension_count,
                                                VkExtensionProperties **extension_properties) {
    VkResult err;
    uint32_t ext_count = 0;
    VkExtensionProperties *ext_ptr = NULL;

    /* repeat get until VK_INCOMPLETE goes away */
    do {
        err = vkEnumerateDeviceExtensionProperties(gpu->obj, layer_name, &ext_count, NULL);
        assert(!err);

        if (ext_ptr) {
            free(ext_ptr);
        }
        ext_ptr = malloc(ext_count * sizeof(VkExtensionProperties));
        err = vkEnumerateDeviceExtensionProperties(gpu->obj, layer_name, &ext_count, ext_ptr);
    } while (err == VK_INCOMPLETE);
    assert(!err);

    *extension_count = ext_count;
    *extension_properties = ext_ptr;
}

static void AppDevInit(struct AppDev *dev, struct AppGpu *gpu) {
    VkDeviceCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueCreateInfoCount = 0,
        .pQueueCreateInfos = NULL,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = NULL,
    };
    VkResult U_ASSERT_ONLY err;

    // Device extensions
    AppGetPhysicalDeviceLayerExtensions(gpu, NULL, &gpu->device_extension_count, &gpu->device_extensions);

    fflush(stdout);

    /* request all queues */
    info.queueCreateInfoCount = gpu->queue_count;
    info.pQueueCreateInfos = gpu->queue_reqs;

    info.enabledLayerCount = 0;
    info.ppEnabledLayerNames = NULL;
    info.enabledExtensionCount = 0;
    info.ppEnabledExtensionNames = NULL;
    dev->gpu = gpu;
    err = vkCreateDevice(gpu->obj, &info, NULL, &dev->obj);
    if (err) ERR_EXIT(err);
}

static void AppDevDestroy(struct AppDev *dev) {
    vkDeviceWaitIdle(dev->obj);
    vkDestroyDevice(dev->obj, NULL);
}

static void AppGetGlobalLayerExtensions(char *layer_name, uint32_t *extension_count, VkExtensionProperties **extension_properties) {
    VkResult err;
    uint32_t ext_count = 0;
    VkExtensionProperties *ext_ptr = NULL;

    /* repeat get until VK_INCOMPLETE goes away */
    do {
        // gets the extension count if the last parameter is NULL
        err = vkEnumerateInstanceExtensionProperties(layer_name, &ext_count, NULL);
        assert(!err);

        if (ext_ptr) {
            free(ext_ptr);
        }
        ext_ptr = malloc(ext_count * sizeof(VkExtensionProperties));
        // gets the extension properties if the last parameter is not NULL
        err = vkEnumerateInstanceExtensionProperties(layer_name, &ext_count, ext_ptr);
    } while (err == VK_INCOMPLETE);
    assert(!err);
    *extension_count = ext_count;
    *extension_properties = ext_ptr;
}

/* Gets a list of layer and instance extensions */
static void AppGetInstanceExtensions(struct AppInstance *inst) {
    VkResult U_ASSERT_ONLY err;

    uint32_t count = 0;

    /* Scan layers */
    VkLayerProperties *global_layer_properties = NULL;
    struct LayerExtensionList *global_layers = NULL;

    do {
        err = vkEnumerateInstanceLayerProperties(&count, NULL);
        assert(!err);

        if (global_layer_properties) {
            free(global_layer_properties);
        }
        global_layer_properties = malloc(sizeof(VkLayerProperties) * count);
        assert(global_layer_properties);

        if (global_layers) {
            free(global_layers);
        }
        global_layers = malloc(sizeof(struct LayerExtensionList) * count);
        assert(global_layers);

        err = vkEnumerateInstanceLayerProperties(&count, global_layer_properties);
    } while (err == VK_INCOMPLETE);
    assert(!err);

    inst->global_layer_count = count;
    inst->global_layers = global_layers;

    for (uint32_t i = 0; i < inst->global_layer_count; i++) {
        VkLayerProperties *src_info = &global_layer_properties[i];
        struct LayerExtensionList *dst_info = &inst->global_layers[i];
        memcpy(&dst_info->layer_properties, src_info, sizeof(VkLayerProperties));

        // Save away layer extension info for report
        // Gets layer extensions, if first parameter is not NULL
        AppGetGlobalLayerExtensions(src_info->layerName, &dst_info->extension_count, &dst_info->extension_properties);
    }
    free(global_layer_properties);

    // Collect global extensions
    inst->global_extension_count = 0;
    // Gets instance extensions, if no layer was specified in the first
    // paramteter
    AppGetGlobalLayerExtensions(NULL, &inst->global_extension_count, &inst->global_extensions);
}

static void AppCreateInstance(struct AppInstance *inst) {
    AppGetInstanceExtensions(inst);

//---Build a list of extensions to load---

    const char *info_instance_extensions[] = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
                                              VK_EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME,
                                              VK_KHR_SURFACE_EXTENSION_NAME,
                                              VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME,
                                              VK_KHR_SHARED_PRESENTABLE_IMAGE_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_WIN32_KHR
                                              VK_KHR_WIN32_SURFACE_EXTENSION_NAME
#elif VK_USE_PLATFORM_XCB_KHR
                                              VK_KHR_XCB_SURFACE_EXTENSION_NAME
#elif VK_USE_PLATFORM_XLIB_KHR
                                              VK_KHR_XLIB_SURFACE_EXTENSION_NAME
#elif VK_USE_PLATFORM_WAYLAND_KHR
                                              VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME
#elif VK_USE_PLATFORM_ANDROID_KHR
                                              VK_KHR_ANDROID_SURFACE_EXTENSION_NAME
#endif
    };
    uint32_t info_instance_extensions_count = ARRAY_SIZE(info_instance_extensions);
    inst->inst_extensions = malloc(sizeof(char *) * ARRAY_SIZE(info_instance_extensions));
    inst->inst_extensions_count = 0;

    for (uint32_t k = 0; (k < info_instance_extensions_count); k++) {
        for (uint32_t j = 0; (j < inst->global_extension_count); j++) {
            const char *found_name = inst->global_extensions[j].extensionName;
            if (!strcmp(info_instance_extensions[k], found_name)) {
                inst->inst_extensions[inst->inst_extensions_count++] = info_instance_extensions[k];
                break;
            }
        }
    }

    //----------------------------------------

    const VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = NULL,
        .pApplicationName = APP_SHORT_NAME,
        .applicationVersion = 1,
        .pEngineName = APP_SHORT_NAME,
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_0,
    };

    VkInstanceCreateInfo inst_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                      .pNext = NULL,
                                      .pApplicationInfo = &app_info,
                                      .enabledLayerCount = 0,
                                      .ppEnabledLayerNames = NULL,
                                      .enabledExtensionCount = inst->inst_extensions_count,
                                      .ppEnabledExtensionNames = inst->inst_extensions};

    VkDebugReportCallbackCreateInfoEXT dbg_info;
    memset(&dbg_info, 0, sizeof(dbg_info));
    dbg_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
    dbg_info.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_INFORMATION_BIT_EXT;
    dbg_info.pfnCallback = DbgCallback;
    inst_info.pNext = &dbg_info;

    VkResult U_ASSERT_ONLY err;
    err = vkCreateInstance(&inst_info, NULL, &inst->instance);
    if (err == VK_ERROR_INCOMPATIBLE_DRIVER) {
        printf("Cannot create Vulkan instance.\n");
        ERR_EXIT(err);
    } else if (err) {
        ERR_EXIT(err);
    }

    inst->vkGetPhysicalDeviceSurfaceSupportKHR =
        (PFN_vkGetPhysicalDeviceSurfaceSupportKHR)vkGetInstanceProcAddr(inst->instance, "vkGetPhysicalDeviceSurfaceSupportKHR");
    inst->vkGetPhysicalDeviceSurfaceCapabilitiesKHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)vkGetInstanceProcAddr(
        inst->instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    inst->vkGetPhysicalDeviceSurfaceFormatsKHR =
        (PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)vkGetInstanceProcAddr(inst->instance, "vkGetPhysicalDeviceSurfaceFormatsKHR");
    inst->vkGetPhysicalDeviceSurfacePresentModesKHR = (PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)vkGetInstanceProcAddr(
        inst->instance, "vkGetPhysicalDeviceSurfacePresentModesKHR");
    inst->vkGetPhysicalDeviceProperties2KHR =
        (PFN_vkGetPhysicalDeviceProperties2KHR)vkGetInstanceProcAddr(inst->instance, "vkGetPhysicalDeviceProperties2KHR");
    inst->vkGetPhysicalDeviceFormatProperties2KHR = (PFN_vkGetPhysicalDeviceFormatProperties2KHR)vkGetInstanceProcAddr(
        inst->instance, "vkGetPhysicalDeviceFormatProperties2KHR");
    inst->vkGetPhysicalDeviceQueueFamilyProperties2KHR = (PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR)vkGetInstanceProcAddr(
        inst->instance, "vkGetPhysicalDeviceQueueFamilyProperties2KHR");
    inst->vkGetPhysicalDeviceFeatures2KHR =
        (PFN_vkGetPhysicalDeviceFeatures2KHR)vkGetInstanceProcAddr(inst->instance, "vkGetPhysicalDeviceFeatures2KHR");
    inst->vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)vkGetInstanceProcAddr(
        inst->instance, "vkGetPhysicalDeviceMemoryProperties2KHR");
    inst->vkGetPhysicalDeviceSurfaceCapabilities2KHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR)vkGetInstanceProcAddr(
        inst->instance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR");
    inst->vkGetPhysicalDeviceSurfaceCapabilities2EXT = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT)vkGetInstanceProcAddr(
        inst->instance, "vkGetPhysicalDeviceSurfaceCapabilities2EXT");
}

//-----------------------------------------------------------

static void AppDestroyInstance(struct AppInstance *inst) {
    free(inst->global_extensions);
    for (uint32_t i = 0; i < inst->global_layer_count; i++) {
        free(inst->global_layers[i].extension_properties);
    }
    free(inst->global_layers);
    free((char**)inst->inst_extensions);
    vkDestroyInstance(inst->instance, NULL);
}

static void AppGpuInit(struct AppGpu *gpu, struct AppInstance *inst, uint32_t id, VkPhysicalDevice obj) {
    uint32_t i;

    memset(gpu, 0, sizeof(*gpu));

    gpu->id = id;
    gpu->obj = obj;
    gpu->inst = inst;

    vkGetPhysicalDeviceProperties(gpu->obj, &gpu->props);

    if (CheckExtensionEnabled(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, gpu->inst->inst_extensions,
                              gpu->inst->inst_extensions_count)) {
        gpu->props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
        gpu->props2.pNext = NULL;

        inst->vkGetPhysicalDeviceProperties2KHR(gpu->obj, &gpu->props2);
    }

    /* get queue count */
    vkGetPhysicalDeviceQueueFamilyProperties(gpu->obj, &gpu->queue_count, NULL);

    gpu->queue_props = malloc(sizeof(gpu->queue_props[0]) * gpu->queue_count);

    if (!gpu->queue_props) ERR_EXIT(VK_ERROR_OUT_OF_HOST_MEMORY);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu->obj, &gpu->queue_count, gpu->queue_props);

    if (CheckExtensionEnabled(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, gpu->inst->inst_extensions,
                              gpu->inst->inst_extensions_count)) {
        gpu->queue_props2 = malloc(sizeof(gpu->queue_props2[0]) * gpu->queue_count);

        if (!gpu->queue_props2) ERR_EXIT(VK_ERROR_OUT_OF_HOST_MEMORY);

        for (i = 0; i < gpu->queue_count; i++) {
            gpu->queue_props2[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR;
            gpu->queue_props2[i].pNext = NULL;
        }

        inst->vkGetPhysicalDeviceQueueFamilyProperties2KHR(gpu->obj, &gpu->queue_count, gpu->queue_props2);
    }

    /* set up queue requests */
    gpu->queue_reqs = malloc(sizeof(*gpu->queue_reqs) * gpu->queue_count);
    if (!gpu->queue_reqs) ERR_EXIT(VK_ERROR_OUT_OF_HOST_MEMORY);
    for (i = 0; i < gpu->queue_count; i++) {
        float *queue_priorities = malloc(gpu->queue_props[i].queueCount * sizeof(float));
        if (!queue_priorities) ERR_EXIT(VK_ERROR_OUT_OF_HOST_MEMORY);
        memset(queue_priorities, 0, gpu->queue_props[i].queueCount * sizeof(float));

        gpu->queue_reqs[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        gpu->queue_reqs[i].pNext = NULL;
        gpu->queue_reqs[i].flags = 0;
        gpu->queue_reqs[i].queueFamilyIndex = i;
        gpu->queue_reqs[i].queueCount = gpu->queue_props[i].queueCount;
        gpu->queue_reqs[i].pQueuePriorities = queue_priorities;
    }

    vkGetPhysicalDeviceMemoryProperties(gpu->obj, &gpu->memory_props);

    vkGetPhysicalDeviceFeatures(gpu->obj, &gpu->features);

    if (CheckExtensionEnabled(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, gpu->inst->inst_extensions,
                              gpu->inst->inst_extensions_count)) {
        gpu->memory_props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR;
        gpu->memory_props2.pNext = NULL;

        inst->vkGetPhysicalDeviceMemoryProperties2KHR(gpu->obj, &gpu->memory_props2);

        gpu->features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
        gpu->features2.pNext = NULL;

        inst->vkGetPhysicalDeviceFeatures2KHR(gpu->obj, &gpu->features2);
    }

    AppDevInit(&gpu->dev, gpu);
    AppDevInitFormats(&gpu->dev);
}

static void AppGpuDestroy(struct AppGpu *gpu) {
    AppDevDestroy(&gpu->dev);
    free(gpu->device_extensions);

    for (uint32_t i = 0; i < gpu->queue_count; i++) {
        free((void *)gpu->queue_reqs[i].pQueuePriorities);
    }
    free(gpu->queue_reqs);

    free(gpu->queue_props);
    if (CheckExtensionEnabled(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, gpu->inst->inst_extensions,
                              gpu->inst->inst_extensions_count)) {
        free(gpu->queue_props2);
    }
}

// clang-format off

//-----------------------------------------------------------

//---------------------------Win32---------------------------
#ifdef VK_USE_PLATFORM_WIN32_KHR

// MS-Windows event handling function:
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}

static void AppCreateWin32Window(struct AppInstance *inst) {
    inst->h_instance = GetModuleHandle(NULL);

    WNDCLASSEX win_class;

    // Initialize the window class structure:
    win_class.cbSize = sizeof(WNDCLASSEX);
    win_class.style = CS_HREDRAW | CS_VREDRAW;
    win_class.lpfnWndProc = WndProc;
    win_class.cbClsExtra = 0;
    win_class.cbWndExtra = 0;
    win_class.hInstance = inst->h_instance;
    win_class.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    win_class.hCursor = LoadCursor(NULL, IDC_ARROW);
    win_class.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    win_class.lpszMenuName = NULL;
    win_class.lpszClassName = APP_SHORT_NAME;
    win_class.hInstance = inst->h_instance;
    win_class.hIconSm = LoadIcon(NULL, IDI_WINLOGO);
    // Register window class:
    if (!RegisterClassEx(&win_class)) {
        // It didn't work, so try to give a useful error:
        printf("Failed to register the window class!\n");
        fflush(stdout);
        exit(1);
    }
    // Create window with the registered class:
    RECT wr = { 0, 0, inst->width, inst->height };
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
    inst->h_wnd = CreateWindowEx(0,
        APP_SHORT_NAME,       // class name
        APP_SHORT_NAME,       // app name
        //WS_VISIBLE | WS_SYSMENU |
        WS_OVERLAPPEDWINDOW,  // window style
        100, 100,             // x/y coords
        wr.right - wr.left,   // width
        wr.bottom - wr.top,   // height
        NULL,                 // handle to parent
        NULL,                 // handle to menu
        inst->h_instance,      // hInstance
        NULL);                // no extra parameters
    if (!inst->h_wnd) {
        // It didn't work, so try to give a useful error:
        printf("Failed to create a window!\n");
        fflush(stdout);
        exit(1);
    }
}

static void AppCreateWin32Surface(struct AppInstance *inst) {
    VkResult U_ASSERT_ONLY err;
    VkWin32SurfaceCreateInfoKHR createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    createInfo.pNext = NULL;
    createInfo.flags = 0;
    createInfo.hinstance = inst->h_instance;
    createInfo.hwnd = inst->h_wnd;
    err = vkCreateWin32SurfaceKHR(inst->instance, &createInfo, NULL, &inst->surface);
    assert(!err);
}

static void AppDestroyWin32Window(struct AppInstance *inst) {
    DestroyWindow(inst->h_wnd);
}
#endif //VK_USE_PLATFORM_WIN32_KHR
//-----------------------------------------------------------

#if defined(VK_USE_PLATFORM_XCB_KHR)     || \
    defined(VK_USE_PLATFORM_XLIB_KHR)    || \
    defined(VK_USE_PLATFORM_WIN32_KHR)
static void AppDestroySurface(struct AppInstance *inst) { //same for all platforms
    vkDestroySurfaceKHR(inst->instance, inst->surface, NULL);
}
#endif

//----------------------------XCB----------------------------

#ifdef VK_USE_PLATFORM_XCB_KHR
static void AppCreateXcbWindow(struct AppInstance *inst) {
    //--Init Connection--
    const xcb_setup_t *setup;
    xcb_screen_iterator_t iter;
    int scr;

    inst->xcb_connection = xcb_connect(NULL, &scr);
    if (inst->xcb_connection == NULL) {
        printf("XCB failed to connect to the X server.\nExiting ...\n");
        fflush(stdout);
        exit(1);
    }

    int conn_error = xcb_connection_has_error(inst->xcb_connection);
    if (conn_error) {
        printf("XCB failed to connect to the X server due to error:%d.\nExiting ...\n", conn_error);
        fflush(stdout);
        exit(1);
    }

    setup = xcb_get_setup(inst->xcb_connection);
    iter = xcb_setup_roots_iterator(setup);
    while (scr-- > 0) {
        xcb_screen_next(&iter);
    }

    inst->xcb_screen = iter.data;
    //-------------------

    inst->xcb_window = xcb_generate_id(inst->xcb_connection);
    xcb_create_window(inst->xcb_connection, XCB_COPY_FROM_PARENT, inst->xcb_window,
                      inst->xcb_screen->root, 0, 0, inst->width, inst->height, 0,
                      XCB_WINDOW_CLASS_INPUT_OUTPUT, inst->xcb_screen->root_visual,
                      0, NULL);

    xcb_intern_atom_cookie_t cookie = xcb_intern_atom(inst->xcb_connection, 1, 12, "WM_PROTOCOLS");
    xcb_intern_atom_reply_t *reply =  xcb_intern_atom_reply(inst->xcb_connection, cookie, 0);
    free(reply);
}

static void AppCreateXcbSurface(struct AppInstance *inst) {
    VkResult U_ASSERT_ONLY err;
    VkXcbSurfaceCreateInfoKHR xcb_createInfo;
    xcb_createInfo.sType      = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
    xcb_createInfo.pNext      = NULL;
    xcb_createInfo.flags      = 0;
    xcb_createInfo.connection = inst->xcb_connection;
    xcb_createInfo.window     = inst->xcb_window;
    err = vkCreateXcbSurfaceKHR(inst->instance, &xcb_createInfo, NULL, &inst->surface);
    assert(!err);
}

static void AppDestroyXcbWindow(struct AppInstance *inst) {
    xcb_destroy_window(inst->xcb_connection, inst->xcb_window);
    xcb_disconnect(inst->xcb_connection);
}
//VK_USE_PLATFORM_XCB_KHR
//-----------------------------------------------------------

//----------------------------XLib---------------------------
#elif VK_USE_PLATFORM_XLIB_KHR
static void AppCreateXlibWindow(struct AppInstance *inst) {
    long visualMask = VisualScreenMask;
    int numberOfVisuals;

    inst->xlib_display = XOpenDisplay(NULL);
    if (inst->xlib_display == NULL) {
        printf("XLib failed to connect to the X server.\nExiting ...\n");
        fflush(stdout);
        exit(1);
    }

    XVisualInfo vInfoTemplate={};
    vInfoTemplate.screen = DefaultScreen(inst->xlib_display);
    XVisualInfo *visualInfo = XGetVisualInfo(inst->xlib_display, visualMask,
                                             &vInfoTemplate, &numberOfVisuals);
    inst->xlib_window = XCreateWindow(
                inst->xlib_display, RootWindow(inst->xlib_display, vInfoTemplate.screen), 0, 0,
                inst->width, inst->height, 0, visualInfo->depth, InputOutput,
                visualInfo->visual, 0, NULL);

    XSync(inst->xlib_display,false);
}

static void AppCreateXlibSurface(struct AppInstance *inst) {
    VkResult U_ASSERT_ONLY err;
    VkXlibSurfaceCreateInfoKHR createInfo;
    createInfo.sType  = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
    createInfo.pNext  = NULL;
    createInfo.flags  = 0;
    createInfo.dpy    = inst->xlib_display;
    createInfo.window = inst->xlib_window;
    err = vkCreateXlibSurfaceKHR(inst->instance, &createInfo, NULL, &inst->surface);
    assert(!err);
}

static void AppDestroyXlibWindow(struct AppInstance *inst) {
    XDestroyWindow(inst->xlib_display, inst->xlib_window);
    XCloseDisplay(inst->xlib_display);
}
#endif //VK_USE_PLATFORM_XLIB_KHR
//-----------------------------------------------------------

#if defined(VK_USE_PLATFORM_XCB_KHR)     || \
    defined(VK_USE_PLATFORM_XLIB_KHR)    || \
    defined(VK_USE_PLATFORM_WIN32_KHR)
static int AppDumpSurfaceFormats(struct AppInstance *inst, struct AppGpu *gpu){
    // Get the list of VkFormat's that are supported:
    VkResult U_ASSERT_ONLY err;
    uint32_t format_count = 0;
    err = inst->vkGetPhysicalDeviceSurfaceFormatsKHR(gpu->obj, inst->surface, &format_count, NULL);
    assert(!err);

    VkSurfaceFormatKHR *surf_formats = (VkSurfaceFormatKHR *)malloc(format_count * sizeof(VkSurfaceFormatKHR));
    if (!surf_formats)
        ERR_EXIT(VK_ERROR_OUT_OF_HOST_MEMORY);
    err = inst->vkGetPhysicalDeviceSurfaceFormatsKHR(gpu->obj, inst->surface, &format_count, surf_formats);
    assert(!err);
    printf("Formats:\t\tcount = %d\n", format_count);

    for (uint32_t i = 0; i < format_count; i++) {
        printf("\t%s\n", VkFormatString(surf_formats[i].format));
    }
    fflush(stdout);

    free(surf_formats);
    return format_count;
}

static int AppDumpSurfacePresentModes(struct AppInstance *inst, struct AppGpu *gpu) {
    // Get the list of VkPresentMode's that are supported:
    VkResult U_ASSERT_ONLY err;
    uint32_t present_mode_count = 0;
    err = inst->vkGetPhysicalDeviceSurfacePresentModesKHR(gpu->obj, inst->surface, &present_mode_count, NULL);
    assert(!err);

    VkPresentModeKHR *surf_present_modes = (VkPresentModeKHR *)malloc(present_mode_count * sizeof(VkPresentInfoKHR));
    if (!surf_present_modes)
        ERR_EXIT(VK_ERROR_OUT_OF_HOST_MEMORY);
    err = inst->vkGetPhysicalDeviceSurfacePresentModesKHR(gpu->obj, inst->surface, &present_mode_count, surf_present_modes);
    assert(!err);
    printf("Present Modes:\t\tcount = %d\n", present_mode_count);

    for (uint32_t i = 0; i < present_mode_count; i++) {
        printf("\t%s\n", VkPresentModeString(surf_present_modes[i]));
    }
    printf("\n");
    fflush(stdout);

    free(surf_present_modes);
    return present_mode_count;
}

static void AppDumpSurfaceCapabilities(struct AppInstance *inst, struct AppGpu *gpu) {
    if (CheckExtensionEnabled(VK_KHR_SURFACE_EXTENSION_NAME, gpu->inst->inst_extensions, gpu->inst->inst_extensions_count)) {

        inst->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu->obj, inst->surface, &inst->surface_capabilities);

        printf("\nVkSurfaceCapabilitiesKHR:\n");
        printf("=========================\n\n");
        printf("\tminImageCount       = %u\n", inst->surface_capabilities.minImageCount);
        printf("\tmaxImageCount       = %u\n", inst->surface_capabilities.maxImageCount);
        printf("\tcurrentExtent:\n");
        printf("\t\twidth       = %u\n", inst->surface_capabilities.currentExtent.width);
        printf("\t\theight      = %u\n", inst->surface_capabilities.currentExtent.height);
        printf("\tminImageExtent:\n");
        printf("\t\twidth       = %u\n", inst->surface_capabilities.minImageExtent.width);
        printf("\t\theight      = %u\n", inst->surface_capabilities.minImageExtent.height);
        printf("\tmaxImageExtent:\n");
        printf("\t\twidth       = %u\n", inst->surface_capabilities.maxImageExtent.width);
        printf("\t\theight      = %u\n", inst->surface_capabilities.maxImageExtent.height);
        printf("\tmaxImageArrayLayers = %u\n", inst->surface_capabilities.maxImageArrayLayers);
        printf("\tsupportedTransform:\n");
        if (inst->surface_capabilities.supportedTransforms == 0) { printf("\t\tNone\n"); }
        if (inst->surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR\n"); }
        if (inst->surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR\n"); }
        if (inst->surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR\n"); }
        if (inst->surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR\n"); }
        if (inst->surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR\n"); }
        if (inst->surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR\n"); }
        if (inst->surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR\n"); }
        if (inst->surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR\n"); }
        if (inst->surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_INHERIT_BIT_KHR\n"); }
        printf("\tcurrentTransform:\n");
        if (inst->surface_capabilities.currentTransform == 0) { printf("\t\tNone\n"); }
        if (inst->surface_capabilities.currentTransform & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR\n"); }
        else if (inst->surface_capabilities.currentTransform & VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR\n"); }
        else if (inst->surface_capabilities.currentTransform & VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR\n"); }
        else if (inst->surface_capabilities.currentTransform & VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR\n"); }
        else if (inst->surface_capabilities.currentTransform & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR\n"); }
        else if (inst->surface_capabilities.currentTransform & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR\n"); }
        else if (inst->surface_capabilities.currentTransform & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR\n"); }
        else if (inst->surface_capabilities.currentTransform & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR\n"); }
        else if (inst->surface_capabilities.currentTransform & VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR) { printf("\t\tVK_SURFACE_TRANSFORM_INHERIT_BIT_KHR\n"); }
        printf("\tsupportedCompositeAlpha:\n");
        if (inst->surface_capabilities.supportedCompositeAlpha == 0) { printf("\t\tNone\n"); }
        if (inst->surface_capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) { printf("\t\tVK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR\n"); }
        if (inst->surface_capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) { printf("\t\tVK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR\n"); }
        if (inst->surface_capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) { printf("\t\tVK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR\n"); }
        if (inst->surface_capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) { printf("\t\tVK_COMPOSITE_ALPHA_INHERIT_BIT_KHR\n"); }
        printf("\tsupportedUsageFlags:\n");
        if (inst->surface_capabilities.supportedUsageFlags == 0) { printf("\t\tNone\n"); }
        if (inst->surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) { printf("\t\tVK_IMAGE_USAGE_TRANSFER_SRC_BIT\n"); }
        if (inst->surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) { printf("\t\tVK_IMAGE_USAGE_TRANSFER_DST_BIT\n"); }
        if (inst->surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_SAMPLED_BIT) { printf("\t\tVK_IMAGE_USAGE_SAMPLED_BIT\n"); }
        if (inst->surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) { printf("\t\tVK_IMAGE_USAGE_STORAGE_BIT\n"); }
        if (inst->surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) { printf("\t\tVK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT\n"); }
        if (inst->surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) { printf("\t\tVK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT\n"); }
        if (inst->surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) { printf("\t\tVK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT\n"); }
        if (inst->surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) { printf("\t\tVK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT\n"); }

        // Get additional surface capability information from vkGetPhysicalDeviceSurfaceCapabilities2EXT
        if (CheckExtensionEnabled(VK_EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME, gpu->inst->inst_extensions, gpu->inst->inst_extensions_count)) {
            memset(&inst->surface_capabilities2_ext, 0, sizeof(VkSurfaceCapabilities2EXT));
            inst->surface_capabilities2_ext.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES2_EXT;
            inst->surface_capabilities2_ext.pNext = NULL;

            inst->vkGetPhysicalDeviceSurfaceCapabilities2EXT(gpu->obj, inst->surface, &inst->surface_capabilities2_ext);

            printf("\nVkSurfaceCapabilities2EXT:\n");
            printf("==========================\n\n");
            printf("\tsupportedSurfaceCounters:\n");
            if (inst->surface_capabilities2_ext.supportedSurfaceCounters == 0) { printf("\t\tNone\n"); }
            if (inst->surface_capabilities2_ext.supportedSurfaceCounters & VK_SURFACE_COUNTER_VBLANK_EXT) { printf("\t\tVK_SURFACE_COUNTER_VBLANK_EXT\n"); }
        }

        // Get additional surface capability information from vkGetPhysicalDeviceSurfaceCapabilities2KHR
        if (CheckExtensionEnabled(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME, gpu->inst->inst_extensions, gpu->inst->inst_extensions_count)) {
            if (CheckExtensionEnabled(VK_KHR_SHARED_PRESENTABLE_IMAGE_EXTENSION_NAME, gpu->inst->inst_extensions, gpu->inst->inst_extensions_count)) {
                inst->shared_surface_capabilities.sType = VK_STRUCTURE_TYPE_SHARED_PRESENT_SURFACE_CAPABILITIES_KHR;
                inst->shared_surface_capabilities.pNext = NULL;
                inst->surface_capabilities2.pNext = &inst->shared_surface_capabilities;
            } else {
                inst->surface_capabilities2.pNext = NULL;
            }

            inst->surface_capabilities2.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR;

            VkPhysicalDeviceSurfaceInfo2KHR surface_info;
            surface_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR;
            surface_info.pNext = NULL;
            surface_info.surface = inst->surface;

            inst->vkGetPhysicalDeviceSurfaceCapabilities2KHR(gpu->obj, &surface_info, &inst->surface_capabilities2);

            void *place = inst->surface_capabilities2.pNext;
            while (place) {
                struct VkStructureHeader* work = (struct VkStructureHeader*) place;
                if (work->sType == VK_STRUCTURE_TYPE_SHARED_PRESENT_SURFACE_CAPABILITIES_KHR) {
                    printf("\nVkSharedPresentSurfaceCapabilitiesKHR:\n");
                    printf("========================================\n");
                    VkSharedPresentSurfaceCapabilitiesKHR* shared_surface_capabilities = (VkSharedPresentSurfaceCapabilitiesKHR*)place;
                    printf("\tsharedPresentSupportedUsageFlags:\n");
                    if (shared_surface_capabilities->sharedPresentSupportedUsageFlags == 0) { printf("\t\tNone\n"); }
                    if (shared_surface_capabilities->sharedPresentSupportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) { printf("\t\tVK_IMAGE_USAGE_TRANSFER_SRC_BIT\n"); }
                    if (shared_surface_capabilities->sharedPresentSupportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) { printf("\t\tVK_IMAGE_USAGE_TRANSFER_DST_BIT\n"); }
                    if (shared_surface_capabilities->sharedPresentSupportedUsageFlags & VK_IMAGE_USAGE_SAMPLED_BIT) { printf("\t\tVK_IMAGE_USAGE_SAMPLED_BIT\n"); }
                    if (shared_surface_capabilities->sharedPresentSupportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) { printf("\t\tVK_IMAGE_USAGE_STORAGE_BIT\n"); }
                    if (shared_surface_capabilities->sharedPresentSupportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) { printf("\t\tVK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT\n"); }
                    if (shared_surface_capabilities->sharedPresentSupportedUsageFlags & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) { printf("\t\tVK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT\n"); }
                    if (shared_surface_capabilities->sharedPresentSupportedUsageFlags & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) { printf("\t\tVK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT\n"); }
                    if (shared_surface_capabilities->sharedPresentSupportedUsageFlags & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) { printf("\t\tVK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT\n"); }
                }

                place = work->pNext;
            }
        }
    }
}

#endif

static void AppDevDumpFormatProps(const struct AppDev *dev, VkFormat fmt)
{
    const VkFormatProperties *props = &dev->format_props[fmt];
    struct {
        const char *name;
        VkFlags flags;
    } features[3];

    features[0].name  = "linearTiling   FormatFeatureFlags";
    features[0].flags = props->linearTilingFeatures;
    features[1].name  = "optimalTiling  FormatFeatureFlags";
    features[1].flags = props->optimalTilingFeatures;
    features[2].name  = "bufferFeatures FormatFeatureFlags";
    features[2].flags = props->bufferFeatures;

    printf("\nFORMAT_%s:", VkFormatString(fmt));
    for (uint32_t i = 0; i < ARRAY_SIZE(features); i++) {
        printf("\n\t%s:", features[i].name);
        if (features[i].flags == 0) {
            printf("\n\t\tNone");
        } else {
            printf("%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s",
               ((features[i].flags & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)                  ? "\n\t\tVK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT"                  : ""),  //0x0001
               ((features[i].flags & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)                  ? "\n\t\tVK_FORMAT_FEATURE_STORAGE_IMAGE_BIT"                  : ""),  //0x0002
               ((features[i].flags & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT)           ? "\n\t\tVK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT"           : ""),  //0x0004
               ((features[i].flags & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT)           ? "\n\t\tVK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT"           : ""),  //0x0008
               ((features[i].flags & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT)           ? "\n\t\tVK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT"           : ""),  //0x0010
               ((features[i].flags & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT)    ? "\n\t\tVK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT"    : ""),  //0x0020
               ((features[i].flags & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)                  ? "\n\t\tVK_FORMAT_FEATURE_VERTEX_BUFFER_BIT"                  : ""),  //0x0040
               ((features[i].flags & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)               ? "\n\t\tVK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT"               : ""),  //0x0080
               ((features[i].flags & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT)         ? "\n\t\tVK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT"         : ""),  //0x0100
               ((features[i].flags & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)       ? "\n\t\tVK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT"       : ""),  //0x0200
               ((features[i].flags & VK_FORMAT_FEATURE_BLIT_SRC_BIT)                       ? "\n\t\tVK_FORMAT_FEATURE_BLIT_SRC_BIT"                       : ""),  //0x0400
               ((features[i].flags & VK_FORMAT_FEATURE_BLIT_DST_BIT)                       ? "\n\t\tVK_FORMAT_FEATURE_BLIT_DST_BIT"                       : ""),  //0x0800
               ((features[i].flags & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)    ? "\n\t\tVK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT"    : ""),  //0x1000
               ((features[i].flags & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG) ? "\n\t\tVK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG" : ""),  //0x2000
               ((features[i].flags & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR)               ? "\n\t\tVK_FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR"               : ""),  //0x4000
               ((features[i].flags & VK_FORMAT_FEATURE_TRANSFER_DST_BIT_KHR)               ? "\n\t\tVK_FORMAT_FEATURE_TRANSFER_DST_BIT_KHR"               : "")); //0x8000
        }
    }
    printf("\n");
}

static void
AppDevDump(const struct AppDev *dev)
{
    printf("Format Properties:\n");
    printf("==================");
    VkFormat fmt;

    for (fmt = 0; fmt < VK_FORMAT_RANGE_SIZE; fmt++) {
        AppDevDumpFormatProps(dev, fmt);
    }
}

#ifdef _WIN32
#define PRINTF_SIZE_T_SPECIFIER    "%Iu"
#else
#define PRINTF_SIZE_T_SPECIFIER    "%zu"
#endif

static void AppGpuDumpFeatures(const struct AppGpu *gpu)
{
    const VkPhysicalDeviceFeatures *features = &gpu->features;

    printf("VkPhysicalDeviceFeatures:\n");
    printf("=========================\n");

    printf("\trobustBufferAccess                      = %u\n", features->robustBufferAccess                     );
    printf("\tfullDrawIndexUint32                     = %u\n", features->fullDrawIndexUint32                    );
    printf("\timageCubeArray                          = %u\n", features->imageCubeArray                         );
    printf("\tindependentBlend                        = %u\n", features->independentBlend                       );
    printf("\tgeometryShader                          = %u\n", features->geometryShader                         );
    printf("\ttessellationShader                      = %u\n", features->tessellationShader                     );
    printf("\tsampleRateShading                       = %u\n", features->sampleRateShading                      );
    printf("\tdualSrcBlend                            = %u\n", features->dualSrcBlend                           );
    printf("\tlogicOp                                 = %u\n", features->logicOp                                );
    printf("\tmultiDrawIndirect                       = %u\n", features->multiDrawIndirect                      );
    printf("\tdrawIndirectFirstInstance               = %u\n", features->drawIndirectFirstInstance              );
    printf("\tdepthClamp                              = %u\n", features->depthClamp                             );
    printf("\tdepthBiasClamp                          = %u\n", features->depthBiasClamp                         );
    printf("\tfillModeNonSolid                        = %u\n", features->fillModeNonSolid                       );
    printf("\tdepthBounds                             = %u\n", features->depthBounds                            );
    printf("\twideLines                               = %u\n", features->wideLines                              );
    printf("\tlargePoints                             = %u\n", features->largePoints                            );
    printf("\talphaToOne                              = %u\n", features->alphaToOne                             );
    printf("\tmultiViewport                           = %u\n", features->multiViewport                          );
    printf("\tsamplerAnisotropy                       = %u\n", features->samplerAnisotropy                      );
    printf("\ttextureCompressionETC2                  = %u\n", features->textureCompressionETC2                 );
    printf("\ttextureCompressionASTC_LDR              = %u\n", features->textureCompressionASTC_LDR             );
    printf("\ttextureCompressionBC                    = %u\n", features->textureCompressionBC                   );
    printf("\tocclusionQueryPrecise                   = %u\n", features->occlusionQueryPrecise                  );
    printf("\tpipelineStatisticsQuery                 = %u\n", features->pipelineStatisticsQuery                );
    printf("\tvertexPipelineStoresAndAtomics          = %u\n", features->vertexPipelineStoresAndAtomics         );
    printf("\tfragmentStoresAndAtomics                = %u\n", features->fragmentStoresAndAtomics               );
    printf("\tshaderTessellationAndGeometryPointSize  = %u\n", features->shaderTessellationAndGeometryPointSize );
    printf("\tshaderImageGatherExtended               = %u\n", features->shaderImageGatherExtended              );
    printf("\tshaderStorageImageExtendedFormats       = %u\n", features->shaderStorageImageExtendedFormats      );
    printf("\tshaderStorageImageMultisample           = %u\n", features->shaderStorageImageMultisample          );
    printf("\tshaderStorageImageReadWithoutFormat     = %u\n", features->shaderStorageImageReadWithoutFormat    );
    printf("\tshaderStorageImageWriteWithoutFormat    = %u\n", features->shaderStorageImageWriteWithoutFormat   );
    printf("\tshaderUniformBufferArrayDynamicIndexing = %u\n", features->shaderUniformBufferArrayDynamicIndexing);
    printf("\tshaderSampledImageArrayDynamicIndexing  = %u\n", features->shaderSampledImageArrayDynamicIndexing );
    printf("\tshaderStorageBufferArrayDynamicIndexing = %u\n", features->shaderStorageBufferArrayDynamicIndexing);
    printf("\tshaderStorageImageArrayDynamicIndexing  = %u\n", features->shaderStorageImageArrayDynamicIndexing );
    printf("\tshaderClipDistance                      = %u\n", features->shaderClipDistance                     );
    printf("\tshaderCullDistance                      = %u\n", features->shaderCullDistance                     );
    printf("\tshaderFloat64                           = %u\n", features->shaderFloat64                          );
    printf("\tshaderInt64                             = %u\n", features->shaderInt64                            );
    printf("\tshaderInt16                             = %u\n", features->shaderInt16                            );
    printf("\tshaderResourceResidency                 = %u\n", features->shaderResourceResidency                );
    printf("\tshaderResourceMinLod                    = %u\n", features->shaderResourceMinLod                   );
    printf("\tsparseBinding                           = %u\n", features->sparseBinding                          );
    printf("\tsparseResidencyBuffer                   = %u\n", features->sparseResidencyBuffer                  );
    printf("\tsparseResidencyImage2D                  = %u\n", features->sparseResidencyImage2D                 );
    printf("\tsparseResidencyImage3D                  = %u\n", features->sparseResidencyImage3D                 );
    printf("\tsparseResidency2Samples                 = %u\n", features->sparseResidency2Samples                );
    printf("\tsparseResidency4Samples                 = %u\n", features->sparseResidency4Samples                );
    printf("\tsparseResidency8Samples                 = %u\n", features->sparseResidency8Samples                );
    printf("\tsparseResidency16Samples                = %u\n", features->sparseResidency16Samples               );
    printf("\tsparseResidencyAliased                  = %u\n", features->sparseResidencyAliased                 );
    printf("\tvariableMultisampleRate                 = %u\n", features->variableMultisampleRate                );
    printf("\tinheritedQueries                        = %u\n", features->inheritedQueries                       );
}

static void AppDumpSparseProps(const VkPhysicalDeviceSparseProperties *sparse_props)
{

    printf("\tVkPhysicalDeviceSparseProperties:\n");
    printf("\t---------------------------------\n");

    printf("\t\tresidencyStandard2DBlockShape            = %u\n", sparse_props->residencyStandard2DBlockShape           );
    printf("\t\tresidencyStandard2DMultisampleBlockShape = %u\n", sparse_props->residencyStandard2DMultisampleBlockShape);
    printf("\t\tresidencyStandard3DBlockShape            = %u\n", sparse_props->residencyStandard3DBlockShape           );
    printf("\t\tresidencyAlignedMipSize                  = %u\n", sparse_props->residencyAlignedMipSize                 );
    printf("\t\tresidencyNonResidentStrict               = %u\n", sparse_props->residencyNonResidentStrict              );
}

static void AppDumpLimits(const VkPhysicalDeviceLimits *limits)
{
    printf("\tVkPhysicalDeviceLimits:\n");
    printf("\t-----------------------\n");
    printf("\t\tmaxImageDimension1D                     = %u\n",                 limits->maxImageDimension1D                    );
    printf("\t\tmaxImageDimension2D                     = %u\n",                 limits->maxImageDimension2D                    );
    printf("\t\tmaxImageDimension3D                     = %u\n",                 limits->maxImageDimension3D                    );
    printf("\t\tmaxImageDimensionCube                   = %u\n",                 limits->maxImageDimensionCube                  );
    printf("\t\tmaxImageArrayLayers                     = %u\n",                 limits->maxImageArrayLayers                    );
    printf("\t\tmaxTexelBufferElements                  = 0x%" PRIxLEAST32 "\n", limits->maxTexelBufferElements                 );
    printf("\t\tmaxUniformBufferRange                   = 0x%" PRIxLEAST32 "\n", limits->maxUniformBufferRange                  );
    printf("\t\tmaxStorageBufferRange                   = 0x%" PRIxLEAST32 "\n", limits->maxStorageBufferRange                  );
    printf("\t\tmaxPushConstantsSize                    = %u\n",                 limits->maxPushConstantsSize                   );
    printf("\t\tmaxMemoryAllocationCount                = %u\n",                 limits->maxMemoryAllocationCount               );
    printf("\t\tmaxSamplerAllocationCount               = %u\n",                 limits->maxSamplerAllocationCount              );
    printf("\t\tbufferImageGranularity                  = 0x%" PRIxLEAST64 "\n", limits->bufferImageGranularity                 );
    printf("\t\tsparseAddressSpaceSize                  = 0x%" PRIxLEAST64 "\n", limits->sparseAddressSpaceSize                 );
    printf("\t\tmaxBoundDescriptorSets                  = %u\n",                 limits->maxBoundDescriptorSets                 );
    printf("\t\tmaxPerStageDescriptorSamplers           = %u\n",                 limits->maxPerStageDescriptorSamplers          );
    printf("\t\tmaxPerStageDescriptorUniformBuffers     = %u\n",                 limits->maxPerStageDescriptorUniformBuffers    );
    printf("\t\tmaxPerStageDescriptorStorageBuffers     = %u\n",                 limits->maxPerStageDescriptorStorageBuffers    );
    printf("\t\tmaxPerStageDescriptorSampledImages      = %u\n",                 limits->maxPerStageDescriptorSampledImages     );
    printf("\t\tmaxPerStageDescriptorStorageImages      = %u\n",                 limits->maxPerStageDescriptorStorageImages     );
    printf("\t\tmaxPerStageDescriptorInputAttachments   = %u\n",                 limits->maxPerStageDescriptorInputAttachments  );
    printf("\t\tmaxPerStageResources                    = %u\n",                 limits->maxPerStageResources                   );
    printf("\t\tmaxDescriptorSetSamplers                = %u\n",                 limits->maxDescriptorSetSamplers               );
    printf("\t\tmaxDescriptorSetUniformBuffers          = %u\n",                 limits->maxDescriptorSetUniformBuffers         );
    printf("\t\tmaxDescriptorSetUniformBuffersDynamic   = %u\n",                 limits->maxDescriptorSetUniformBuffersDynamic  );
    printf("\t\tmaxDescriptorSetStorageBuffers          = %u\n",                 limits->maxDescriptorSetStorageBuffers         );
    printf("\t\tmaxDescriptorSetStorageBuffersDynamic   = %u\n",                 limits->maxDescriptorSetStorageBuffersDynamic  );
    printf("\t\tmaxDescriptorSetSampledImages           = %u\n",                 limits->maxDescriptorSetSampledImages          );
    printf("\t\tmaxDescriptorSetStorageImages           = %u\n",                 limits->maxDescriptorSetStorageImages          );
    printf("\t\tmaxDescriptorSetInputAttachments        = %u\n",                 limits->maxDescriptorSetInputAttachments       );
    printf("\t\tmaxVertexInputAttributes                = %u\n",                 limits->maxVertexInputAttributes               );
    printf("\t\tmaxVertexInputBindings                  = %u\n",                 limits->maxVertexInputBindings                 );
    printf("\t\tmaxVertexInputAttributeOffset           = 0x%" PRIxLEAST32 "\n", limits->maxVertexInputAttributeOffset          );
    printf("\t\tmaxVertexInputBindingStride             = 0x%" PRIxLEAST32 "\n", limits->maxVertexInputBindingStride            );
    printf("\t\tmaxVertexOutputComponents               = %u\n",                 limits->maxVertexOutputComponents              );
    printf("\t\tmaxTessellationGenerationLevel          = %u\n",                 limits->maxTessellationGenerationLevel         );
    printf("\t\tmaxTessellationPatchSize                        = %u\n",                 limits->maxTessellationPatchSize                       );
    printf("\t\tmaxTessellationControlPerVertexInputComponents  = %u\n",                 limits->maxTessellationControlPerVertexInputComponents );
    printf("\t\tmaxTessellationControlPerVertexOutputComponents = %u\n",                 limits->maxTessellationControlPerVertexOutputComponents);
    printf("\t\tmaxTessellationControlPerPatchOutputComponents  = %u\n",                 limits->maxTessellationControlPerPatchOutputComponents );
    printf("\t\tmaxTessellationControlTotalOutputComponents     = %u\n",                 limits->maxTessellationControlTotalOutputComponents    );
    printf("\t\tmaxTessellationEvaluationInputComponents        = %u\n",                 limits->maxTessellationEvaluationInputComponents       );
    printf("\t\tmaxTessellationEvaluationOutputComponents       = %u\n",                 limits->maxTessellationEvaluationOutputComponents      );
    printf("\t\tmaxGeometryShaderInvocations            = %u\n",                 limits->maxGeometryShaderInvocations           );
    printf("\t\tmaxGeometryInputComponents              = %u\n",                 limits->maxGeometryInputComponents             );
    printf("\t\tmaxGeometryOutputComponents             = %u\n",                 limits->maxGeometryOutputComponents            );
    printf("\t\tmaxGeometryOutputVertices               = %u\n",                 limits->maxGeometryOutputVertices              );
    printf("\t\tmaxGeometryTotalOutputComponents        = %u\n",                 limits->maxGeometryTotalOutputComponents       );
    printf("\t\tmaxFragmentInputComponents              = %u\n",                 limits->maxFragmentInputComponents             );
    printf("\t\tmaxFragmentOutputAttachments            = %u\n",                 limits->maxFragmentOutputAttachments           );
    printf("\t\tmaxFragmentDualSrcAttachments           = %u\n",                 limits->maxFragmentDualSrcAttachments          );
    printf("\t\tmaxFragmentCombinedOutputResources      = %u\n",                 limits->maxFragmentCombinedOutputResources     );
    printf("\t\tmaxComputeSharedMemorySize              = 0x%" PRIxLEAST32 "\n", limits->maxComputeSharedMemorySize             );
    printf("\t\tmaxComputeWorkGroupCount[0]             = %u\n",                 limits->maxComputeWorkGroupCount[0]            );
    printf("\t\tmaxComputeWorkGroupCount[1]             = %u\n",                 limits->maxComputeWorkGroupCount[1]            );
    printf("\t\tmaxComputeWorkGroupCount[2]             = %u\n",                 limits->maxComputeWorkGroupCount[2]            );
    printf("\t\tmaxComputeWorkGroupInvocations          = %u\n",                 limits->maxComputeWorkGroupInvocations         );
    printf("\t\tmaxComputeWorkGroupSize[0]              = %u\n",                 limits->maxComputeWorkGroupSize[0]             );
    printf("\t\tmaxComputeWorkGroupSize[1]              = %u\n",                 limits->maxComputeWorkGroupSize[1]             );
    printf("\t\tmaxComputeWorkGroupSize[2]              = %u\n",                 limits->maxComputeWorkGroupSize[2]             );
    printf("\t\tsubPixelPrecisionBits                   = %u\n",                 limits->subPixelPrecisionBits                  );
    printf("\t\tsubTexelPrecisionBits                   = %u\n",                 limits->subTexelPrecisionBits                  );
    printf("\t\tmipmapPrecisionBits                     = %u\n",                 limits->mipmapPrecisionBits                    );
    printf("\t\tmaxDrawIndexedIndexValue                = %u\n",                 limits->maxDrawIndexedIndexValue               );
    printf("\t\tmaxDrawIndirectCount                    = %u\n",                 limits->maxDrawIndirectCount                   );
    printf("\t\tmaxSamplerLodBias                       = %f\n",                 limits->maxSamplerLodBias                      );
    printf("\t\tmaxSamplerAnisotropy                    = %f\n",                 limits->maxSamplerAnisotropy                   );
    printf("\t\tmaxViewports                            = %u\n",                 limits->maxViewports                           );
    printf("\t\tmaxViewportDimensions[0]                = %u\n",                 limits->maxViewportDimensions[0]               );
    printf("\t\tmaxViewportDimensions[1]                = %u\n",                 limits->maxViewportDimensions[1]               );
    printf("\t\tviewportBoundsRange[0]                  =%13f\n",                 limits->viewportBoundsRange[0]                 );
    printf("\t\tviewportBoundsRange[1]                  =%13f\n",                 limits->viewportBoundsRange[1]                 );
    printf("\t\tviewportSubPixelBits                    = %u\n",                 limits->viewportSubPixelBits                   );
    printf("\t\tminMemoryMapAlignment                   = " PRINTF_SIZE_T_SPECIFIER "\n", limits->minMemoryMapAlignment         );
    printf("\t\tminTexelBufferOffsetAlignment           = 0x%" PRIxLEAST64 "\n", limits->minTexelBufferOffsetAlignment          );
    printf("\t\tminUniformBufferOffsetAlignment         = 0x%" PRIxLEAST64 "\n", limits->minUniformBufferOffsetAlignment        );
    printf("\t\tminStorageBufferOffsetAlignment         = 0x%" PRIxLEAST64 "\n", limits->minStorageBufferOffsetAlignment        );
    printf("\t\tminTexelOffset                          =%3d\n",                 limits->minTexelOffset                         );
    printf("\t\tmaxTexelOffset                          =%3d\n",                 limits->maxTexelOffset                         );
    printf("\t\tminTexelGatherOffset                    =%3d\n",                 limits->minTexelGatherOffset                   );
    printf("\t\tmaxTexelGatherOffset                    =%3d\n",                 limits->maxTexelGatherOffset                   );
    printf("\t\tminInterpolationOffset                  =%9f\n",                 limits->minInterpolationOffset                 );
    printf("\t\tmaxInterpolationOffset                  =%9f\n",                 limits->maxInterpolationOffset                 );
    printf("\t\tsubPixelInterpolationOffsetBits         = %u\n",                 limits->subPixelInterpolationOffsetBits        );
    printf("\t\tmaxFramebufferWidth                     = %u\n",                 limits->maxFramebufferWidth                    );
    printf("\t\tmaxFramebufferHeight                    = %u\n",                 limits->maxFramebufferHeight                   );
    printf("\t\tmaxFramebufferLayers                    = %u\n",                 limits->maxFramebufferLayers                   );
    printf("\t\tframebufferColorSampleCounts            = %u\n",                 limits->framebufferColorSampleCounts           );
    printf("\t\tframebufferDepthSampleCounts            = %u\n",                 limits->framebufferDepthSampleCounts           );
    printf("\t\tframebufferStencilSampleCounts          = %u\n",                 limits->framebufferStencilSampleCounts         );
    printf("\t\tframebufferNoAttachmentsSampleCounts    = %u\n",                 limits->framebufferNoAttachmentsSampleCounts   );
    printf("\t\tmaxColorAttachments                     = %u\n",                 limits->maxColorAttachments                    );
    printf("\t\tsampledImageColorSampleCounts           = %u\n",                 limits->sampledImageColorSampleCounts          );
    printf("\t\tsampledImageDepthSampleCounts           = %u\n",                 limits->sampledImageDepthSampleCounts          );
    printf("\t\tsampledImageStencilSampleCounts         = %u\n",                 limits->sampledImageStencilSampleCounts        );
    printf("\t\tsampledImageIntegerSampleCounts         = %u\n",                 limits->sampledImageIntegerSampleCounts        );
    printf("\t\tstorageImageSampleCounts                = %u\n",                 limits->storageImageSampleCounts               );
    printf("\t\tmaxSampleMaskWords                      = %u\n",                 limits->maxSampleMaskWords                     );
    printf("\t\ttimestampComputeAndGraphics             = %u\n",                 limits->timestampComputeAndGraphics            );
    printf("\t\ttimestampPeriod                         = %f\n",                 limits->timestampPeriod                        );
    printf("\t\tmaxClipDistances                        = %u\n",                 limits->maxClipDistances                       );
    printf("\t\tmaxCullDistances                        = %u\n",                 limits->maxCullDistances                       );
    printf("\t\tmaxCombinedClipAndCullDistances         = %u\n",                 limits->maxCombinedClipAndCullDistances        );
    printf("\t\tdiscreteQueuePriorities                 = %u\n",                 limits->discreteQueuePriorities                );
    printf("\t\tpointSizeRange[0]                       = %f\n",                 limits->pointSizeRange[0]                      );
    printf("\t\tpointSizeRange[1]                       = %f\n",                 limits->pointSizeRange[1]                      );
    printf("\t\tlineWidthRange[0]                       = %f\n",                 limits->lineWidthRange[0]                      );
    printf("\t\tlineWidthRange[1]                       = %f\n",                 limits->lineWidthRange[1]                      );
    printf("\t\tpointSizeGranularity                    = %f\n",                 limits->pointSizeGranularity                   );
    printf("\t\tlineWidthGranularity                    = %f\n",                 limits->lineWidthGranularity                   );
    printf("\t\tstrictLines                             = %u\n",                 limits->strictLines                            );
    printf("\t\tstandardSampleLocations                 = %u\n",                 limits->standardSampleLocations                );
    printf("\t\toptimalBufferCopyOffsetAlignment        = 0x%" PRIxLEAST64 "\n", limits->optimalBufferCopyOffsetAlignment       );
    printf("\t\toptimalBufferCopyRowPitchAlignment      = 0x%" PRIxLEAST64 "\n", limits->optimalBufferCopyRowPitchAlignment     );
    printf("\t\tnonCoherentAtomSize                     = 0x%" PRIxLEAST64 "\n", limits->nonCoherentAtomSize                    );
}

static void AppGpuDumpProps(const struct AppGpu *gpu)
{
    const VkPhysicalDeviceProperties *props = &gpu->props;
    const uint32_t apiVersion=props->apiVersion;
    const uint32_t major = VK_VERSION_MAJOR(apiVersion);
    const uint32_t minor = VK_VERSION_MINOR(apiVersion);
    const uint32_t patch = VK_VERSION_PATCH(apiVersion);

    printf("VkPhysicalDeviceProperties:\n");
    printf("===========================\n");
    printf("\tapiVersion     = 0x%" PRIxLEAST32 "  (%d.%d.%d)\n", apiVersion, major, minor, patch);
    printf("\tdriverVersion  = %u (0x%" PRIxLEAST32 ")\n",props->driverVersion, props->driverVersion);
    printf("\tvendorID       = 0x%04x\n",                 props->vendorID);
    printf("\tdeviceID       = 0x%04x\n",                 props->deviceID);
    printf("\tdeviceType     = %s\n",                     VkPhysicalDeviceTypeString(props->deviceType));
    printf("\tdeviceName     = %s\n",                     props->deviceName);

    AppDumpLimits(&gpu->props.limits);
    AppDumpSparseProps(&gpu->props.sparseProperties);

    fflush(stdout);
}
// clang-format on

static void AppDumpExtensions(const char *indent, const char *layer_name, const uint32_t extension_count,
                              const VkExtensionProperties *extension_properties) {
    uint32_t i;
    if (layer_name && (strlen(layer_name) > 0)) {
        printf("%s%s Extensions", indent, layer_name);
    } else {
        printf("%sExtensions", indent);
    }
    printf("\tcount = %d\n", extension_count);
    for (i = 0; i < extension_count; i++) {
        VkExtensionProperties const *ext_prop = &extension_properties[i];

        printf("%s\t", indent);
        printf("%-36s: extension revision %2d\n", ext_prop->extensionName, ext_prop->specVersion);
    }
    fflush(stdout);
}

static void AppGpuDumpQueueProps(const struct AppGpu *gpu, uint32_t id) {
    const VkQueueFamilyProperties *props = &gpu->queue_props[id];

    printf("VkQueueFamilyProperties[%d]:\n", id);
    printf("===========================\n");
    char *sep = "";  // separator character
    printf("\tqueueFlags         = ");
    if (props->queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        printf("GRAPHICS");
        sep = " | ";
    }
    if (props->queueFlags & VK_QUEUE_COMPUTE_BIT) {
        printf("%sCOMPUTE", sep);
        sep = " | ";
    }
    if (props->queueFlags & VK_QUEUE_TRANSFER_BIT) {
        printf("%sTRANSFER", sep);
        sep = " | ";
    }
    if (props->queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) {
        printf("%sSPARSE", sep);
    }
    printf("\n");

    printf("\tqueueCount         = %u\n", props->queueCount);
    printf("\ttimestampValidBits = %u\n", props->timestampValidBits);
    printf("\tminImageTransferGranularity = (%d, %d, %d)\n", props->minImageTransferGranularity.width,
           props->minImageTransferGranularity.height, props->minImageTransferGranularity.depth);
    fflush(stdout);
}

// This prints a number of bytes in a human-readable format according to prefixes of the International System of Quantities (ISQ),
// defined in ISO/IEC 80000. The prefixes used here are not SI prefixes, but rather the binary prefixes based on powers of 1024
// (kibi-, mebi-, gibi- etc.).
#define kBufferSize 32

static char *HumanReadable(const size_t sz) {
    const char prefixes[] = "KMGTPEZY";
    char buf[kBufferSize];
    int which = -1;
    double result = (double)sz;
    while (result > 1024 && which < 7) {
        result /= 1024;
        ++which;
    }

    char unit[] = "\0i";
    if (which >= 0) {
        unit[0] = prefixes[which];
    }
    snprintf(buf, kBufferSize, "%.2f %sB", result, unit);
    return strndup(buf, kBufferSize);
}

static void AppGpuDumpMemoryProps(const struct AppGpu *gpu) {
    const VkPhysicalDeviceMemoryProperties *props = &gpu->memory_props;

    printf("VkPhysicalDeviceMemoryProperties:\n");
    printf("=================================\n");
    printf("\tmemoryTypeCount       = %u\n", props->memoryTypeCount);
    for (uint32_t i = 0; i < props->memoryTypeCount; i++) {
        printf("\tmemoryTypes[%u] :\n", i);
        printf("\t\theapIndex     = %u\n", props->memoryTypes[i].heapIndex);
        printf("\t\tpropertyFlags = 0x%" PRIxLEAST32 ":\n", props->memoryTypes[i].propertyFlags);

        // Print each named flag, if it is set.
        VkFlags flags = props->memoryTypes[i].propertyFlags;
#define PRINT_FLAG(FLAG) \
    if (flags & FLAG) printf("\t\t\t" #FLAG "\n");
        PRINT_FLAG(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        PRINT_FLAG(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        PRINT_FLAG(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        PRINT_FLAG(VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        PRINT_FLAG(VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
#undef PRINT_FLAG
    }
    printf("\n");
    printf("\tmemoryHeapCount       = %u\n", props->memoryHeapCount);
    for (uint32_t i = 0; i < props->memoryHeapCount; i++) {
        printf("\tmemoryHeaps[%u] :\n", i);
        const VkDeviceSize memSize = props->memoryHeaps[i].size;
        char *mem_size_human_readable = HumanReadable((const size_t)memSize);
        printf("\t\tsize          = " PRINTF_SIZE_T_SPECIFIER " (0x%" PRIxLEAST64 ") (%s)\n", (size_t)memSize, memSize,
               mem_size_human_readable);
        free(mem_size_human_readable);

        VkMemoryHeapFlags heap_flags = props->memoryHeaps[i].flags;
        printf("\t\tflags:\n\t\t\t");
        printf((heap_flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "VK_MEMORY_HEAP_DEVICE_LOCAL_BIT\n" : "None\n");
    }
    fflush(stdout);
}

static void AppGpuDump(const struct AppGpu *gpu) {
    uint32_t i;

    printf("\nDevice Properties and Extensions :\n");
    printf("==================================\n");
    printf("GPU%u\n", gpu->id);
    AppGpuDumpProps(gpu);
    printf("\n");
    AppDumpExtensions("", "Device", gpu->device_extension_count, gpu->device_extensions);
    printf("\n");
    for (i = 0; i < gpu->queue_count; i++) {
        AppGpuDumpQueueProps(gpu, i);
        printf("\n");
    }
    AppGpuDumpMemoryProps(gpu);
    printf("\n");
    AppGpuDumpFeatures(gpu);
    printf("\n");
    AppDevDump(&gpu->dev);
}

#ifdef _WIN32
// Enlarges the console window to have a large scrollback size.
static void ConsoleEnlarge() {
    HANDLE console_handle = GetStdHandle(STD_OUTPUT_HANDLE);

    // make the console window bigger
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    COORD buffer_size;
    if (GetConsoleScreenBufferInfo(console_handle, &csbi)) {
        buffer_size.X = csbi.dwSize.X + 30;
        buffer_size.Y = 20000;
        SetConsoleScreenBufferSize(console_handle, buffer_size);
    }

    SMALL_RECT r;
    r.Left = r.Top = 0;
    r.Right = csbi.dwSize.X - 1 + 30;
    r.Bottom = 50;
    SetConsoleWindowInfo(console_handle, true, &r);

    // change the console window title
    SetConsoleTitle(TEXT(APP_SHORT_NAME));
}
#endif

int main(int argc, char **argv) {
    uint32_t vulkan_major, vulkan_minor, vulkan_patch;
    struct AppGpu *gpus;
    VkPhysicalDevice *objs;
    uint32_t gpu_count;
    VkResult err;
    struct AppInstance inst;

#ifdef _WIN32
    if (ConsoleIsExclusive()) ConsoleEnlarge();
#endif

    vulkan_major = VK_VERSION_MAJOR(VK_API_VERSION_1_0);
    vulkan_minor = VK_VERSION_MINOR(VK_API_VERSION_1_0);
    vulkan_patch = VK_VERSION_PATCH(VK_HEADER_VERSION);

    printf("===========\n");
    printf("VULKAN INFO\n");
    printf("===========\n\n");
    printf("Vulkan API Version: %d.%d.%d\n\n", vulkan_major, vulkan_minor, vulkan_patch);

    AppCreateInstance(&inst);

    printf("\nInstance Extensions:\n");
    printf("====================\n");
    AppDumpExtensions("", "Instance", inst.global_extension_count, inst.global_extensions);

    err = vkEnumeratePhysicalDevices(inst.instance, &gpu_count, NULL);
    if (err) ERR_EXIT(err);
    objs = malloc(sizeof(objs[0]) * gpu_count);
    if (!objs) ERR_EXIT(VK_ERROR_OUT_OF_HOST_MEMORY);
    err = vkEnumeratePhysicalDevices(inst.instance, &gpu_count, objs);
    if (err) ERR_EXIT(err);

    gpus = malloc(sizeof(gpus[0]) * gpu_count);
    if (!gpus) ERR_EXIT(VK_ERROR_OUT_OF_HOST_MEMORY);
    for (uint32_t i = 0; i < gpu_count; i++) {
        AppGpuInit(&gpus[i], &inst, i, objs[i]);
        printf("\n\n");
    }

    //---Layer-Device-Extensions---
    printf("Layers: count = %d\n", inst.global_layer_count);
    printf("=======\n");
    for (uint32_t i = 0; i < inst.global_layer_count; i++) {
        uint32_t layer_major, layer_minor, layer_patch;
        char spec_version[64], layer_version[64];
        VkLayerProperties const *layer_prop = &inst.global_layers[i].layer_properties;

        ExtractVersion(layer_prop->specVersion, &layer_major, &layer_minor, &layer_patch);
        snprintf(spec_version, sizeof(spec_version), "%d.%d.%d", layer_major, layer_minor, layer_patch);
        snprintf(layer_version, sizeof(layer_version), "%d", layer_prop->implementationVersion);
        printf("%s (%s) Vulkan version %s, layer version %s\n", layer_prop->layerName, (char *)layer_prop->description,
               spec_version, layer_version);

        AppDumpExtensions("\t", "Layer", inst.global_layers[i].extension_count, inst.global_layers[i].extension_properties);

        char *layer_name = inst.global_layers[i].layer_properties.layerName;
        printf("\tDevices \tcount = %d\n", gpu_count);
        for (uint32_t j = 0; j < gpu_count; j++) {
            printf("\t\tGPU id       : %u (%s)\n", j, gpus[j].props.deviceName);
            uint32_t count = 0;
            VkExtensionProperties *props;
            AppGetPhysicalDeviceLayerExtensions(&gpus[j], layer_name, &count, &props);
            AppDumpExtensions("\t\t", "Layer-Device", count, props);
            free(props);
        }
        printf("\n");
    }
    fflush(stdout);
    //-----------------------------

    printf("Presentable Surfaces:\n");
    printf("=====================\n");
    inst.width = 256;
    inst.height = 256;
    int format_count = 0;
    int present_mode_count = 0;

#if defined(VK_USE_PLATFORM_XCB_KHR) || defined(VK_USE_PLATFORM_XLIB_KHR)
    if (getenv("DISPLAY") == NULL) {
        printf("'DISPLAY' environment variable not set... Exiting!\n");
        fflush(stdout);
        exit(1);
    }
#endif
//--WIN32--
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (CheckExtensionEnabled(VK_KHR_WIN32_SURFACE_EXTENSION_NAME, inst.inst_extensions, inst.inst_extensions_count)) {
        AppCreateWin32Window(&inst);
        for (uint32_t i = 0; i < gpu_count; i++) {
            AppCreateWin32Surface(&inst);
            printf("GPU id       : %u (%s)\n", i, gpus[i].props.deviceName);
            printf("Surface type : %s\n", VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
            format_count += AppDumpSurfaceFormats(&inst, &gpus[i]);
            present_mode_count += AppDumpSurfacePresentModes(&inst, &gpus[i]);
            AppDumpSurfaceCapabilities(&inst, &gpus[i]);
            AppDestroySurface(&inst);
        }
        AppDestroyWin32Window(&inst);
    }
//--XCB--
#elif VK_USE_PLATFORM_XCB_KHR
    if (CheckExtensionEnabled(VK_KHR_XCB_SURFACE_EXTENSION_NAME, inst.inst_extensions, inst.inst_extensions_count)) {
        AppCreateXcbWindow(&inst);
        for (uint32_t i = 0; i < gpu_count; i++) {
            AppCreateXcbSurface(&inst);
            printf("GPU id       : %u (%s)\n", i, gpus[i].props.deviceName);
            printf("Surface type : %s\n", VK_KHR_XCB_SURFACE_EXTENSION_NAME);
            format_count += AppDumpSurfaceFormats(&inst, &gpus[i]);
            present_mode_count += AppDumpSurfacePresentModes(&inst, &gpus[i]);
            AppDumpSurfaceCapabilities(&inst, &gpus[i]);
            AppDestroySurface(&inst);
        }
        AppDestroyXcbWindow(&inst);
    }
//--XLIB--
#elif VK_USE_PLATFORM_XLIB_KHR
    if (CheckExtensionEnabled(VK_KHR_XLIB_SURFACE_EXTENSION_NAME, inst.inst_extensions, inst.inst_extensions_count)) {
        AppCreateXlibWindow(&inst);
        for (uint32_t i = 0; i < gpu_count; i++) {
            AppCreateXlibSurface(&inst);
            printf("GPU id       : %u (%s)\n", i, gpus[i].props.deviceName);
            printf("Surface type : %s\n", VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
            format_count += AppDumpSurfaceFormats(&inst, &gpus[i]);
            present_mode_count += AppDumpSurfacePresentModes(&inst, &gpus[i]);
            AppDumpSurfaceCapabilities(&inst, &gpus[i]);
            AppDestroySurface(&inst);
        }
        AppDestroyXlibWindow(&inst);
    }
#endif
    // TODO: Android / Wayland / MIR
    if (!format_count && !present_mode_count) printf("None found\n");
    //---------

    for (uint32_t i = 0; i < gpu_count; i++) {
        AppGpuDump(&gpus[i]);
        printf("\n\n");
    }

    for (uint32_t i = 0; i < gpu_count; i++) AppGpuDestroy(&gpus[i]);
    free(gpus);
    free(objs);

    AppDestroyInstance(&inst);

    fflush(stdout);
#ifdef _WIN32
    if (ConsoleIsExclusive()) Sleep(INFINITE);
#endif

    return 0;
}
