/*
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
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
 * Author: Cody Northrop <cody@lunarg.com>
 * Author: David Pinedo <david@lunarg.com>
 * Author: Jon Ashburn <jon@lunarg.com>
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <list>
#include <map>
#include <set>
#include <vector>
#include <fstream>

using namespace std;

#include "vk_dispatch_table_helper.h"
#include "vk_layer_config.h"
#include "vk_layer_table.h"
#include "vk_layer_extension_utils.h"
#include "vk_layer_utils.h"

#include "screenshot_parsing.h"

#ifdef ANDROID

#include <android/log.h>
#include <sys/system_properties.h>

static char android_env[64] = {};
const char *env_var = "debug.vulkan.screenshot";
const char *env_var_old = env_var;
#else  //Linux or Windows
const char *env_var_old = "_VK_SCREENSHOT";
const char *env_var = "VK_SCREENSHOT_FRAMES";
const char *env_var_format = "VK_SCREENSHOT_FORMAT";
#endif

#ifdef ANDROID
char *android_exec(const char *cmd) {
    FILE *pipe = popen(cmd, "r");
    if (pipe != nullptr) {
        fgets(android_env, 64, pipe);
        pclose(pipe);
    }

    // Only if the value is set will we get a string back
    if (strlen(android_env) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "screenshot", "Vulkan screenshot layer capturing: %s", android_env);
        return android_env;
    }

    return nullptr;
}

char *android_getenv(const char *key) {
    std::string command("getprop ");
    command += key;
    return android_exec(command.c_str());
}

static inline char *local_getenv(const char *name) { return android_getenv(name); }

static inline void local_free_getenv(const char *val) {}

#elif defined(__linux__)
static inline char *local_getenv(const char *name) { return getenv(name); }

static inline void local_free_getenv(const char *val) {}

#elif defined(_WIN32)

static inline char *local_getenv(const char *name) {
    char *retVal;
    DWORD valSize;

    valSize = GetEnvironmentVariableA(name, NULL, 0);

    // valSize DOES include the null terminator, so for any set variable
    // will always be at least 1. If it's 0, the variable wasn't set.
    if (valSize == 0) return NULL;

    // TODO; FIXME This should be using any app defined memory allocation
    retVal = (char *)malloc(valSize);

    GetEnvironmentVariableA(name, retVal, valSize);

    return retVal;
}

static inline void local_free_getenv(const char *val) { free((void *)val); }
#endif

namespace screenshot {

static int globalLockInitialized = 0;
static loader_platform_thread_mutex globalLock;

const char *vk_screenshot_format = nullptr;

bool printFormatWarning = true;

typedef enum colorSpaceFormat {
    UNDEFINED = 0,
    UNORM = 1,
    SNORM = 2,
    USCALED = 3,
    SSCALED = 4,
    UINT = 5,
    SINT = 6,
    SRGB = 7
} colorSpaceFormat;

colorSpaceFormat userColorSpaceFormat = UNDEFINED;

// unordered map: associates a swap chain with a device, image extent, format,
// and list of images
typedef struct {
    VkDevice device;
    VkExtent2D imageExtent;
    VkFormat format;
    VkImage *imageList;
} SwapchainMapStruct;
static unordered_map<VkSwapchainKHR, SwapchainMapStruct *> swapchainMap;

// unordered map: associates an image with a device, image extent, and format
typedef struct {
    VkDevice device;
    VkExtent2D imageExtent;
    VkFormat format;
} ImageMapStruct;
static unordered_map<VkImage, ImageMapStruct *> imageMap;

// unordered map: associates a device with a queue, commandPool, and physical
// device also contains per device info including dispatch table
typedef struct {
    VkLayerDispatchTable *device_dispatch_table;
    bool wsi_enabled;
    VkQueue queue;
    std::list<VkCommandPool> commandPools;
    VkPhysicalDevice physicalDevice;
    PFN_vkSetDeviceLoaderData pfn_dev_init;
} DeviceMapStruct;
static unordered_map<VkDevice, DeviceMapStruct *> deviceMap;

// unordered map: associates a physical device with an instance
typedef struct { VkInstance instance; } PhysDeviceMapStruct;
static unordered_map<VkPhysicalDevice, PhysDeviceMapStruct *> physDeviceMap;

// set: list of frames to take screenshots without duplication.
static set<int> screenshotFrames;

// Flag indicating we have received the frame list
static bool screenshotFramesReceived = false;

// Screenshots will be generated from screenShotFrameRange's startFrame to startFrame+count-1 with skipped Interval in between.
static FrameRange screenShotFrameRange = {false, 0, SCREEN_SHOT_FRAMES_UNLIMITED, SCREEN_SHOT_FRAMES_INTERVAL_DEFAULT};

// Get maximum frame number of the frame range
// FrameRange* pFrameRange, the specified frame rang
// return:
//  maximum frame number of the frame range,
//  if it's unlimited range, the return will be SCREEN_SHOT_FRAMES_UNLIMITED
static int getEndFrameOfRange(FrameRange *pFrameRange) {
    int endFrameOfRange = SCREEN_SHOT_FRAMES_UNLIMITED;
    if (pFrameRange->count != SCREEN_SHOT_FRAMES_UNLIMITED) {
        endFrameOfRange = pFrameRange->startFrame + (pFrameRange->count - 1) * pFrameRange->interval;
    }
    return endFrameOfRange;
}

// detect if frameNumber is in the range of pFrameRange, also detect if frameNumber is a frame on which a screenshot should be
// generated.
// int frameNumber, the frame number.
// FrameRange* pFrameRange, the specified frame range.
// bool *pScreenShotFrame, if pScreenShotFrame is not nullptr, indicate(return) if frameNumber is a frame on which a screenshot
// should be generated.
// return:
//  if frameNumber is in the range of pFrameRange.
static bool isInScreenShotFrameRange(int frameNumber, FrameRange *pFrameRange, bool *pScreenShotFrame) {
    bool inRange = false, screenShotFrame = false;
    if (pFrameRange->valid) {
        if (pFrameRange->count != SCREEN_SHOT_FRAMES_UNLIMITED) {
            int endFrame = getEndFrameOfRange(pFrameRange);
            if ((frameNumber >= pFrameRange->startFrame) &&
                ((frameNumber <= endFrame) || (endFrame == SCREEN_SHOT_FRAMES_UNLIMITED))) {
                inRange = true;
            }
        } else {
            inRange = true;
        }
        if (inRange) {
            screenShotFrame = (((frameNumber - pFrameRange->startFrame) % pFrameRange->interval) == 0);
        }
    }
    if (pScreenShotFrame != nullptr) {
        *pScreenShotFrame = screenShotFrame;
    }
    return inRange;
}

//Get users request is specific color space format required
void readScreenShotFormatENV(void) {
#ifndef ANDROID
    vk_screenshot_format = local_getenv(env_var_format);
#endif
    if (vk_screenshot_format && *vk_screenshot_format) {
        if (!strcmp(vk_screenshot_format, "UNORM")) {
            userColorSpaceFormat = UNORM;
        } else if (!strcmp(vk_screenshot_format, "SRGB")) {
            userColorSpaceFormat = SRGB;
        } else if (!strcmp(vk_screenshot_format, "SNORM")) {
            userColorSpaceFormat = SNORM;
        } else if (!strcmp(vk_screenshot_format, "USCALED")) {
            userColorSpaceFormat = USCALED;
        } else if (!strcmp(vk_screenshot_format, "SSCALED")) {
            userColorSpaceFormat = SSCALED;
        } else if (!strcmp(vk_screenshot_format, "UINT")) {
            userColorSpaceFormat = UINT;
        } else if (!strcmp(vk_screenshot_format, "SINT")) {
            userColorSpaceFormat = SINT;
        } else {
#ifdef ANDROID
#else
            fprintf(stderr, "Selected format:%s\nIs NOT in the list:\nUNORM, SNORM, USCALED, SSCALED, UINT, SINT, SRGB\n"
                            "Swapchain Colorspace will be used instead\n", vk_screenshot_format);
#endif
        }
    }
}

// detect if frameNumber reach or beyond the right edge for screenshot in the range.
// return:
//       if frameNumber is already the last screenshot frame of the range(mean no another screenshot frame number >frameNumber and
//       just in the range)
//       if the range is invalid, return true.
static bool isEndOfScreenShotFrameRange(int frameNumber, FrameRange *pFrameRange) {
    bool endOfScreenShotFrameRange = false, screenShotFrame = false;
    if (!pFrameRange->valid) {
        endOfScreenShotFrameRange = true;
    } else {
        int endFrame = getEndFrameOfRange(pFrameRange);
        if (endFrame != SCREEN_SHOT_FRAMES_UNLIMITED) {
            if (isInScreenShotFrameRange(frameNumber, pFrameRange, &screenShotFrame)) {
                if ((frameNumber >= endFrame) && screenShotFrame) {
                    endOfScreenShotFrameRange = true;
                }
            }
        }
    }
    return endOfScreenShotFrameRange;
}

// Parse comma-separated frame list string into the set
static void populate_frame_list(const char *vk_screenshot_frames) {
    string spec(vk_screenshot_frames), word;
    size_t start = 0, comma = 0;

    if (!isOptionBelongToScreenShotRange(vk_screenshot_frames)) {
        while (start < spec.size()) {
            int frameToAdd;
            comma = spec.find(',', start);
            if (comma == string::npos)
                word = string(spec, start);
            else
                word = string(spec, start, comma - start);
            frameToAdd = atoi(word.c_str());
            // Add the frame number to set, but only do it if the word
            // started with a digit and if
            // it's not already in the list
            if (*(word.c_str()) >= '0' && *(word.c_str()) <= '9') {
                screenshotFrames.insert(frameToAdd);
            }
            if (comma == string::npos) break;
            start = comma + 1;
        }
    } else {
        int parsingStatus = initScreenShotFrameRange(vk_screenshot_frames, &screenShotFrameRange);
        if (parsingStatus != 0) {
            fprintf(stderr, "Screenshot range error\n");
        }
    }

    screenshotFramesReceived = true;
}

static bool memory_type_from_properties(VkPhysicalDeviceMemoryProperties *memory_properties, uint32_t typeBits,
                                        VkFlags requirements_mask, uint32_t *typeIndex) {
    // Search memtypes to find first index with those properties
    for (uint32_t i = 0; i < 32; i++) {
        if ((typeBits & 1) == 1) {
            // Type is available, does it match user properties?
            if ((memory_properties->memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
                *typeIndex = i;
                return true;
            }
        }
        typeBits >>= 1;
    }
    // No memory types matched, return failure
    return false;
}

static DeviceMapStruct *get_dev_info(VkDevice dev) {
    auto it = deviceMap.find(dev);
    if (it == deviceMap.end())
        return NULL;
    else
        return it->second;
}

static void init_screenshot() {
    if (!globalLockInitialized) {
        // TODO/TBD: Need to delete this mutex sometime.  How???  One
        // suggestion is to call this during vkCreateInstance(), and then we
        // can clean it up during vkDestroyInstance().  However, that requires
        // that the layer have per-instance locks.  We need to come back and
        // address this soon.
        loader_platform_thread_create_mutex(&globalLock);
        globalLockInitialized = 1;
    }
    readScreenShotFormatENV();
}

// Track allocated resources in writePPM()
// and clean them up when they go out of scope.
struct WritePPMCleanupData {
    VkDevice device;
    VkLayerDispatchTable *pTableDevice;
    VkImage image2;
    VkImage image3;
    VkDeviceMemory mem2;
    VkDeviceMemory mem3;
    bool mem2mapped;
    bool mem3mapped;
    VkCommandBuffer commandBuffer;
    VkCommandPool commandPool;
    ~WritePPMCleanupData();
};

WritePPMCleanupData::~WritePPMCleanupData() {
    if (mem2mapped) pTableDevice->UnmapMemory(device, mem2);
    if (mem2) pTableDevice->FreeMemory(device, mem2, NULL);
    if (image2) pTableDevice->DestroyImage(device, image2, NULL);

    if (mem3mapped) pTableDevice->UnmapMemory(device, mem3);
    if (mem3) pTableDevice->FreeMemory(device, mem3, NULL);
    if (image3) pTableDevice->DestroyImage(device, image3, NULL);

    if (commandBuffer) pTableDevice->FreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

// Save an image to a PPM image file.
//
// This function issues commands to copy/convert the swapchain image
// from whatever compatible format the swapchain image uses
// to a single format (VK_FORMAT_R8G8B8A8_UNORM) so that the converted
// result can be easily written to a PPM file.
//
// Error handling: If there is a problem, this function should silently
// fail without affecting the Present operation going on in the caller.
// The numerous debug asserts are to catch programming errors and are not
// expected to assert.  Recovery and clean up are implemented for image memory
// allocation failures.
// (TODO) It would be nice to pass any failure info to DebugReport or something.
static void writePPM(const char *filename, VkImage image1) {
    VkResult err;
    bool pass;

    // Bail immediately if we can't find the image.
    if (imageMap.empty() || imageMap.find(image1) == imageMap.end()) return;

    // Collect object info from maps.  This info is generally recorded
    // by the other functions hooked in this layer.
    VkDevice device = imageMap[image1]->device;
    VkPhysicalDevice physicalDevice = deviceMap[device]->physicalDevice;
    VkInstance instance = physDeviceMap[physicalDevice]->instance;
    VkQueue queue = deviceMap[device]->queue;
    DeviceMapStruct *devMap = get_dev_info(device);
    if (NULL == devMap) {
        assert(0);
        return;
    }
    VkLayerDispatchTable *pTableDevice = devMap->device_dispatch_table;
    VkLayerDispatchTable *pTableQueue = get_dev_info(static_cast<VkDevice>(static_cast<void *>(queue)))->device_dispatch_table;
    VkLayerInstanceDispatchTable *pInstanceTable;
    pInstanceTable = instance_dispatch_table(instance);

    // Gather incoming image info and check image format for compatibility with
    // the target format.
    // This function supports both 24-bit and 32-bit swapchain images.
    uint32_t const width = imageMap[image1]->imageExtent.width;
    uint32_t const height = imageMap[image1]->imageExtent.height;
    VkFormat const format = imageMap[image1]->format;
    uint32_t const numChannels = FormatChannelCount(format);

    if ((3 != numChannels) && (4 != numChannels)) {
        assert(0);
        return;
    }

    // Initial dest format is undefined as we will look for one
    VkFormat destformat = VK_FORMAT_UNDEFINED;

    //This variable set by readScreenShotFormatENV func during init
    if (userColorSpaceFormat != UNDEFINED) {
        switch (userColorSpaceFormat) {
            case UNORM:
                if (numChannels == 4)
                    destformat = VK_FORMAT_R8G8B8A8_UNORM;
                else
                    destformat = VK_FORMAT_R8G8B8_UNORM;
                break;
            case SRGB:
                if (numChannels == 4)
                    destformat = VK_FORMAT_R8G8B8A8_SRGB;
                else
                    destformat = VK_FORMAT_R8G8B8_SRGB;
                break;
            case SNORM:
                if (numChannels == 4)
                    destformat = VK_FORMAT_R8G8B8A8_SNORM;
                else
                    destformat = VK_FORMAT_R8G8B8_SNORM;
                break;
            case USCALED:
                if (numChannels == 4)
                    destformat = VK_FORMAT_R8G8B8A8_USCALED;
                else
                    destformat = VK_FORMAT_R8G8B8_USCALED;
                break;
            case SSCALED:
                if (numChannels == 4)
                    destformat = VK_FORMAT_R8G8B8A8_SSCALED;
                else
                    destformat = VK_FORMAT_R8G8B8_SSCALED;
                break;
            case UINT:
                if (numChannels == 4)
                    destformat = VK_FORMAT_R8G8B8A8_UINT;
                else
                    destformat = VK_FORMAT_R8G8B8_UINT;
                break;
            case SINT:
                if (numChannels == 4)
                    destformat = VK_FORMAT_R8G8B8A8_SINT;
                else
                    destformat = VK_FORMAT_R8G8B8_SINT;
                break;
            default:
                destformat = VK_FORMAT_UNDEFINED;
                break;
        }
    }

    // User did not require sepecific format so we use same colorspace with
    // swapchain format
    if (destformat == VK_FORMAT_UNDEFINED) {
        // Here we reserve swapchain color space only as RGBA swizzle will be later.
        //
        // One Potential optimization here would be: set destination to RGB all the
        // time instead RGBA. PPM does not support Alpha channel, so we can write
        // RGB one row by row but RGBA written one pixel at a time.
        // This requires BLIT operation to get involved but current drivers (mostly)
        // does not support BLIT operations on 3 Channel rendertargets.
        // So format conversion gets costly.
        if (numChannels == 4) {
            if (FormatIsUNorm(format))
                destformat = VK_FORMAT_R8G8B8A8_UNORM;
            else if (FormatIsSRGB(format))
                destformat = VK_FORMAT_R8G8B8A8_SRGB;
            else if (FormatIsSNorm(format))
                destformat = VK_FORMAT_R8G8B8A8_SNORM;
            else if (FormatIsUScaled(format))
                destformat = VK_FORMAT_R8G8B8A8_USCALED;
            else if (FormatIsSScaled(format))
                destformat = VK_FORMAT_R8G8B8A8_SSCALED;
            else if (FormatIsUInt(format))
                destformat = VK_FORMAT_R8G8B8A8_UINT;
            else if (FormatIsSInt(format))
                destformat = VK_FORMAT_R8G8B8A8_SINT;
        } else { //numChannels 3
            if (FormatIsUNorm(format))
                destformat = VK_FORMAT_R8G8B8_UNORM;
            else if (FormatIsSRGB(format))
                destformat = VK_FORMAT_R8G8B8_SRGB;
            else if (FormatIsSNorm(format))
                destformat = VK_FORMAT_R8G8B8_SNORM;
            else if (FormatIsUScaled(format))
                destformat = VK_FORMAT_R8G8B8_USCALED;
            else if (FormatIsSScaled(format))
                destformat = VK_FORMAT_R8G8B8_SSCALED;
            else if (FormatIsUInt(format))
                destformat = VK_FORMAT_R8G8B8_UINT;
            else if (FormatIsSInt(format))
                destformat = VK_FORMAT_R8G8B8_SINT;
        }
    }

    //Still could not find the right format then we use UNORM
    if (destformat == VK_FORMAT_UNDEFINED)
    {
#ifdef ANDROID
#else
        if (printFormatWarning) {
            fprintf(stderr, "Swapchain format is not in the list:\nUNORM, SNORM, USCALED, SSCALED, UINT, SINT, SRGB\n"
                            "UNORM colorspace will be used instead\n");
            printFormatWarning = false;
        }
#endif
        if (numChannels == 4)
            destformat = VK_FORMAT_R8G8B8A8_UNORM;
        else
            destformat = VK_FORMAT_R8G8B8_UNORM;
    }

    if ((FormatCompatibilityClass(destformat) != FormatCompatibilityClass(format))) {
        assert(0);
        return;
    }

    // General Approach
    //
    // The idea here is to copy/convert the swapchain image into another image
    // that can be mapped and read by the CPU to produce a PPM file.
    // The image must be untiled and converted to a specific format for easy
    // parsing.  The memory for the final image must be host-visible.
    // Note that in Vulkan, a BLIT operation must be used to perform a format
    // conversion.
    //
    // Devices vary in their ability to blit to/from linear and optimal tiling.
    // So we must query the device properties to get this information.
    //
    // If the device cannot BLIT to a LINEAR image, then the operation must be
    // done in two steps:
    // 1) BLIT the swapchain image (image1) to a temp image (image2) that is
    // created with TILING_OPTIMAL.
    // 2) COPY image2 to another temp image (image3) that is created with
    // TILING_LINEAR.
    // 3) Map image 3 and write the PPM file.
    //
    // If the device can BLIT to a LINEAR image, then:
    // 1) BLIT the swapchain image (image1) to a temp image (image2) that is
    // created with TILING_LINEAR.
    // 2) Map image 2 and write the PPM file.
    //
    // There seems to be no way to tell if the swapchain image (image1) is tiled
    // or not.  We therefore assume that the BLIT operation can always read from
    // both linear and optimal tiled (swapchain) images.
    // There is therefore no point in looking at the BLIT_SRC properties.
    //
    // There is also the optimization where the incoming and target formats are
    // the same.  In this case, just do a COPY.

    VkFormatProperties targetFormatProps;
    pInstanceTable->GetPhysicalDeviceFormatProperties(physicalDevice, destformat, &targetFormatProps);
    bool need2steps = false;
    bool copyOnly = false;
    if (destformat == format) {
        copyOnly = true;
    } else {
        bool const bltLinear = targetFormatProps.linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT ? true : false;
        bool const bltOptimal = targetFormatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT ? true : false;
        if (!bltLinear && !bltOptimal) {
            // Cannot blit to either target tiling type.  It should be pretty
            // unlikely to have a device that cannot blit to either type.
            // But punt by just doing a copy and possibly have the wrong
            // colors.  This should be quite rare.
            copyOnly = true;
        } else if (!bltLinear && bltOptimal) {
            // Cannot blit to a linear target but can blt to optimal, so copy
            // after blit is needed.
            need2steps = true;
        }
        // Else bltLinear is available and only 1 step is needed.
    }

    // Put resources that need to be cleaned up in a struct with a destructor
    // so that things get cleaned up when this function is exited.
    WritePPMCleanupData data = {};
    data.device = device;
    data.pTableDevice = pTableDevice;

    // Set up the image creation info for both the blit and copy images, in case
    // both are needed.
    VkImageCreateInfo imgCreateInfo2 = {
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        NULL,
        0,
        VK_IMAGE_TYPE_2D,
        destformat,
        {width, height, 1},
        1,
        1,
        VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_LINEAR,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        NULL,
        VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImageCreateInfo imgCreateInfo3 = imgCreateInfo2;

    // If we need both images, set up image2 to be read/write and tiled.
    if (need2steps) {
        imgCreateInfo2.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgCreateInfo2.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    VkMemoryAllocateInfo memAllocInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL,
        0,  // allocationSize, queried later
        0   // memoryTypeIndex, queried later
    };
    VkMemoryRequirements memRequirements;
    VkPhysicalDeviceMemoryProperties memoryProperties;

    // Create image2 and allocate its memory.  It could be the intermediate or
    // final image.
    err = pTableDevice->CreateImage(device, &imgCreateInfo2, NULL, &data.image2);
    assert(!err);
    if (VK_SUCCESS != err) return;
    pTableDevice->GetImageMemoryRequirements(device, data.image2, &memRequirements);
    memAllocInfo.allocationSize = memRequirements.size;
    pInstanceTable->GetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    pass = memory_type_from_properties(&memoryProperties, memRequirements.memoryTypeBits,
                                       need2steps ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                       &memAllocInfo.memoryTypeIndex);
    assert(pass);
    err = pTableDevice->AllocateMemory(device, &memAllocInfo, NULL, &data.mem2);
    assert(!err);
    if (VK_SUCCESS != err) return;
    err = pTableQueue->BindImageMemory(device, data.image2, data.mem2, 0);
    assert(!err);
    if (VK_SUCCESS != err) return;

    // Create image3 and allocate its memory, if needed.
    if (need2steps) {
        err = pTableDevice->CreateImage(device, &imgCreateInfo3, NULL, &data.image3);
        assert(!err);
        if (VK_SUCCESS != err) return;
        pTableDevice->GetImageMemoryRequirements(device, data.image3, &memRequirements);
        memAllocInfo.allocationSize = memRequirements.size;
        pInstanceTable->GetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        pass = memory_type_from_properties(&memoryProperties, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                           &memAllocInfo.memoryTypeIndex);
        assert(pass);
        err = pTableDevice->AllocateMemory(device, &memAllocInfo, NULL, &data.mem3);
        assert(!err);
        if (VK_SUCCESS != err) return;
        err = pTableQueue->BindImageMemory(device, data.image3, data.mem3, 0);
        assert(!err);
        if (VK_SUCCESS != err) return;
    }

    // Set up the command buffer.  We get a command buffer from a pool we saved
    // in a hooked function, which would be the application's pool.
    if (deviceMap[device]->commandPools.empty()) {
        assert(!deviceMap[device]->commandPools.empty());
        return;
    }

    VkCommandPool commandPool = deviceMap[device]->commandPools.front();

    const VkCommandBufferAllocateInfo allocCommandBufferInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL,
        commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    data.commandPool = commandPool;
    err = pTableDevice->AllocateCommandBuffers(device, &allocCommandBufferInfo, &data.commandBuffer);
    assert(!err);
    if (VK_SUCCESS != err) return;

    VkDevice cmdBuf = static_cast<VkDevice>(static_cast<void *>(data.commandBuffer));
    deviceMap.emplace(cmdBuf, devMap);
    VkLayerDispatchTable *pTableCommandBuffer;
    pTableCommandBuffer = get_dev_info(cmdBuf)->device_dispatch_table;

    // We have just created a dispatchable object, but the dispatch table has
    // not been placed in the object yet.  When a "normal" application creates
    // a command buffer, the dispatch table is installed by the top-level api
    // binding (trampoline.c). But here, we have to do it ourselves.
    if (!devMap->pfn_dev_init) {
        *((const void **)data.commandBuffer) = *(void **)device;
    } else {
        err = devMap->pfn_dev_init(device, (void *)data.commandBuffer);
        assert(!err);
    }

    const VkCommandBufferBeginInfo commandBufferBeginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    err = pTableCommandBuffer->BeginCommandBuffer(data.commandBuffer, &commandBufferBeginInfo);
    assert(!err);

    // This barrier is used to transition from/to present Layout
    VkImageMemoryBarrier presentMemoryBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                                 NULL,
                                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                                                 VK_ACCESS_TRANSFER_READ_BIT,
                                                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                 VK_QUEUE_FAMILY_IGNORED,
                                                 VK_QUEUE_FAMILY_IGNORED,
                                                 image1,
                                                 {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};

    // This barrier is used to transition from a newly-created layout to a blt
    // or copy destination layout.
    VkImageMemoryBarrier destMemoryBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                              NULL,
                                              0,
                                              VK_ACCESS_TRANSFER_WRITE_BIT,
                                              VK_IMAGE_LAYOUT_UNDEFINED,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                              VK_QUEUE_FAMILY_IGNORED,
                                              VK_QUEUE_FAMILY_IGNORED,
                                              data.image2,
                                              {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};

    // This barrier is used to transition a dest layout to general layout.
    VkImageMemoryBarrier generalMemoryBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                                 NULL,
                                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                                                 VK_ACCESS_TRANSFER_READ_BIT,
                                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                 VK_IMAGE_LAYOUT_GENERAL,
                                                 VK_QUEUE_FAMILY_IGNORED,
                                                 VK_QUEUE_FAMILY_IGNORED,
                                                 data.image2,
                                                 {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};

    VkPipelineStageFlags srcStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags dstStages = VK_PIPELINE_STAGE_TRANSFER_BIT;

    // The source image needs to be transitioned from present to transfer
    // source.
    pTableCommandBuffer->CmdPipelineBarrier(data.commandBuffer, srcStages, dstStages, 0, 0, NULL, 0, NULL, 1,
                                            &presentMemoryBarrier);

    // image2 needs to be transitioned from its undefined state to transfer
    // destination.
    pTableCommandBuffer->CmdPipelineBarrier(data.commandBuffer, srcStages, dstStages, 0, 0, NULL, 0, NULL, 1, &destMemoryBarrier);

    const VkImageCopy imageCopyRegion = {
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {width, height, 1}};

    if (copyOnly) {
        pTableCommandBuffer->CmdCopyImage(data.commandBuffer, image1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, data.image2,
                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopyRegion);
    } else {
        VkImageBlit imageBlitRegion = {};
        imageBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlitRegion.srcSubresource.baseArrayLayer = 0;
        imageBlitRegion.srcSubresource.layerCount = 1;
        imageBlitRegion.srcSubresource.mipLevel = 0;
        imageBlitRegion.srcOffsets[1].x = width;
        imageBlitRegion.srcOffsets[1].y = height;
        imageBlitRegion.srcOffsets[1].z = 1;
        imageBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlitRegion.dstSubresource.baseArrayLayer = 0;
        imageBlitRegion.dstSubresource.layerCount = 1;
        imageBlitRegion.dstSubresource.mipLevel = 0;
        imageBlitRegion.dstOffsets[1].x = width;
        imageBlitRegion.dstOffsets[1].y = height;
        imageBlitRegion.dstOffsets[1].z = 1;

        pTableCommandBuffer->CmdBlitImage(data.commandBuffer, image1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, data.image2,
                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageBlitRegion, VK_FILTER_NEAREST);
        if (need2steps) {
            // image 3 needs to be transitioned from its undefined state to a
            // transfer destination.
            destMemoryBarrier.image = data.image3;
            pTableCommandBuffer->CmdPipelineBarrier(data.commandBuffer, srcStages, dstStages, 0, 0, NULL, 0, NULL, 1,
                                                    &destMemoryBarrier);

            // Transition image2 so that it can be read for the upcoming copy to
            // image 3.
            destMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            destMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            destMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            destMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            destMemoryBarrier.image = data.image2;
            pTableCommandBuffer->CmdPipelineBarrier(data.commandBuffer, srcStages, dstStages, 0, 0, NULL, 0, NULL, 1,
                                                    &destMemoryBarrier);

            // This step essentially untiles the image.
            pTableCommandBuffer->CmdCopyImage(data.commandBuffer, data.image2, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, data.image3,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopyRegion);
            generalMemoryBarrier.image = data.image3;
        }
    }

    // The destination needs to be transitioned from the optimal copy format to
    // the format we can read with the CPU.
    pTableCommandBuffer->CmdPipelineBarrier(data.commandBuffer, srcStages, dstStages, 0, 0, NULL, 0, NULL, 1,
                                            &generalMemoryBarrier);

    // Restore the swap chain image layout to what it was before.
    // This may not be strictly needed, but it is generally good to restore
    // things to original state.
    presentMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    presentMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    presentMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    presentMemoryBarrier.dstAccessMask = 0;
    pTableCommandBuffer->CmdPipelineBarrier(data.commandBuffer, srcStages, dstStages, 0, 0, NULL, 0, NULL, 1,
                                            &presentMemoryBarrier);

    err = pTableCommandBuffer->EndCommandBuffer(data.commandBuffer);
    assert(!err);

    VkFence nullFence = {VK_NULL_HANDLE};
    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = NULL;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = NULL;
    submitInfo.pWaitDstStageMask = NULL;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &data.commandBuffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = NULL;

    err = pTableQueue->QueueSubmit(queue, 1, &submitInfo, nullFence);
    assert(!err);

    err = pTableQueue->QueueWaitIdle(queue);
    assert(!err);

    err = pTableDevice->DeviceWaitIdle(device);
    assert(!err);

    // Map the final image so that the CPU can read it.
    const VkImageSubresource sr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkSubresourceLayout srLayout;
    const char *ptr;
    if (!need2steps) {
        pTableDevice->GetImageSubresourceLayout(device, data.image2, &sr, &srLayout);
        err = pTableDevice->MapMemory(device, data.mem2, 0, VK_WHOLE_SIZE, 0, (void **)&ptr);
        assert(!err);
        if (VK_SUCCESS != err) return;
        data.mem2mapped = true;
    } else {
        pTableDevice->GetImageSubresourceLayout(device, data.image3, &sr, &srLayout);
        err = pTableDevice->MapMemory(device, data.mem3, 0, VK_WHOLE_SIZE, 0, (void **)&ptr);
        assert(!err);
        if (VK_SUCCESS != err) return;
        data.mem3mapped = true;
    }

    // Write the data to a PPM file.
    ofstream file(filename, ios::binary);
    assert(file.is_open());

    if (!file.is_open()) {
#ifdef ANDROID
        __android_log_print(ANDROID_LOG_DEBUG, "screenshot",
                            "Failed to open output file: %s.  Be sure to grant read and write permissions.", filename);
#else
        fprintf(stderr, "Failed to open output file:%s,  Be sure to grant read and write permissions\n", filename);
#endif
        return;
    }

    file << "P6\n";
    file << width << "\n";
    file << height << "\n";
    file << 255 << "\n";

    ptr += srLayout.offset;
    if (3 == numChannels) {
        for (uint32_t y = 0; y < height; y++) {
            file.write(ptr, 3 * width);
            ptr += srLayout.rowPitch;
        }
    } else if (4 == numChannels) {
        for (uint32_t y = 0; y < height; y++) {
            const unsigned int *row = (const unsigned int *)ptr;
            for (uint32_t x = 0; x < width; x++) {
                file.write((char *)row, 3);
                row++;
            }
            ptr += srLayout.rowPitch;
        }
    }
    file.close();

    // Clean up handled by ~WritePPMCleanupData()
}

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                                              VkInstance *pInstance) {
    VkLayerInstanceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);

    assert(chain_info->u.pLayerInfo);
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    assert(fpGetInstanceProcAddr);
    PFN_vkCreateInstance fpCreateInstance = (PFN_vkCreateInstance)fpGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    if (fpCreateInstance == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = fpCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS) return result;

    initInstanceTable(*pInstance, fpGetInstanceProcAddr);

    init_screenshot();

    return result;
}

// TODO hook DestroyInstance to cleanup

static void createDeviceRegisterExtensions(const VkDeviceCreateInfo *pCreateInfo, VkDevice device) {
    uint32_t i;
    DeviceMapStruct *devMap = get_dev_info(device);
    VkLayerDispatchTable *pDisp = devMap->device_dispatch_table;
    PFN_vkGetDeviceProcAddr gpa = pDisp->GetDeviceProcAddr;
    pDisp->CreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)gpa(device, "vkCreateSwapchainKHR");
    pDisp->GetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)gpa(device, "vkGetSwapchainImagesKHR");
    pDisp->AcquireNextImageKHR = (PFN_vkAcquireNextImageKHR)gpa(device, "vkAcquireNextImageKHR");
    pDisp->QueuePresentKHR = (PFN_vkQueuePresentKHR)gpa(device, "vkQueuePresentKHR");
    devMap->wsi_enabled = false;
    for (i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        if (strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) devMap->wsi_enabled = true;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice gpu, const VkDeviceCreateInfo *pCreateInfo,
                                            const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
    VkLayerDeviceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);

    assert(chain_info->u.pLayerInfo);
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    VkInstance instance = physDeviceMap[gpu]->instance;
    PFN_vkCreateDevice fpCreateDevice = (PFN_vkCreateDevice)fpGetInstanceProcAddr(instance, "vkCreateDevice");
    if (fpCreateDevice == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = fpCreateDevice(gpu, pCreateInfo, pAllocator, pDevice);
    if (result != VK_SUCCESS) {
        return result;
    }

    assert(deviceMap.find(*pDevice) == deviceMap.end());
    DeviceMapStruct *deviceMapElem = new DeviceMapStruct;
    deviceMap[*pDevice] = deviceMapElem;

    // Setup device dispatch table
    deviceMapElem->device_dispatch_table = new VkLayerDispatchTable;
    layer_init_device_dispatch_table(*pDevice, deviceMapElem->device_dispatch_table, fpGetDeviceProcAddr);

    createDeviceRegisterExtensions(pCreateInfo, *pDevice);
    // Create a mapping from a device to a physicalDevice
    deviceMapElem->physicalDevice = gpu;

    // store the loader callback for initializing created dispatchable objects
    chain_info = get_chain_info(pCreateInfo, VK_LOADER_DATA_CALLBACK);
    if (chain_info) {
        deviceMapElem->pfn_dev_init = chain_info->u.pfnSetDeviceLoaderData;
    } else {
        deviceMapElem->pfn_dev_init = NULL;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(VkInstance instance, uint32_t *pPhysicalDeviceCount,
                                                        VkPhysicalDevice *pPhysicalDevices) {
    VkResult result;

    VkLayerInstanceDispatchTable *pTable = instance_dispatch_table(instance);
    result = pTable->EnumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices);
    if (result == VK_SUCCESS && *pPhysicalDeviceCount > 0 && pPhysicalDevices) {
        for (uint32_t i = 0; i < *pPhysicalDeviceCount; i++) {
            // Create a mapping from a physicalDevice to an instance
            if (physDeviceMap[pPhysicalDevices[i]] == NULL) {
                PhysDeviceMapStruct *physDeviceMapElem = new PhysDeviceMapStruct;
                physDeviceMap[pPhysicalDevices[i]] = physDeviceMapElem;
            }
            physDeviceMap[pPhysicalDevices[i]]->instance = instance;
        }
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    DeviceMapStruct *devMap = get_dev_info(device);
    assert(devMap);
    VkLayerDispatchTable *pDisp = devMap->device_dispatch_table;
    pDisp->DestroyDevice(device, pAllocator);

    local_free_getenv(vk_screenshot_format);
    loader_platform_thread_lock_mutex(&globalLock);
    delete pDisp;
    delete devMap;

    deviceMap.erase(device);
    loader_platform_thread_unlock_mutex(&globalLock);
}

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice device, uint32_t queueNodeIndex, uint32_t queueIndex, VkQueue *pQueue) {
    DeviceMapStruct *devMap = get_dev_info(device);
    assert(devMap);
    VkLayerDispatchTable *pDisp = devMap->device_dispatch_table;
    pDisp->GetDeviceQueue(device, queueNodeIndex, queueIndex, pQueue);

    // Save the device queue in a map if we are taking screenshots.
    loader_platform_thread_lock_mutex(&globalLock);
    if (screenshotFramesReceived && screenshotFrames.empty() && !screenShotFrameRange.valid) {
        // No screenshots in the list to take
        loader_platform_thread_unlock_mutex(&globalLock);
        return;
    }

    VkDevice que = static_cast<VkDevice>(static_cast<void *>(*pQueue));
    deviceMap.emplace(que, devMap);

    // Create a mapping from a device to a queue
    devMap->queue = *pQueue;
    loader_platform_thread_unlock_mutex(&globalLock);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo *pCreateInfo,
                                                 const VkAllocationCallbacks *pAllocator, VkCommandPool *pCommandPool) {
    DeviceMapStruct *devMap = get_dev_info(device);
    assert(devMap);
    VkLayerDispatchTable *pDisp = devMap->device_dispatch_table;
    VkResult result = pDisp->CreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);

    // Save the command pool on a map if we are taking screenshots.
    loader_platform_thread_lock_mutex(&globalLock);
    if (screenshotFramesReceived && screenshotFrames.empty() && !screenShotFrameRange.valid) {
        // No screenshots in the list to take
        loader_platform_thread_unlock_mutex(&globalLock);
        return result;
    }

    // Create a mapping from a device to a commandPool
    devMap->commandPools.push_front(*pCommandPool);
    loader_platform_thread_unlock_mutex(&globalLock);
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks *pAllocator) {
    DeviceMapStruct *devMap = get_dev_info(device);
    assert(devMap);
    VkLayerDispatchTable *pDisp = devMap->device_dispatch_table;
    pDisp->DestroyCommandPool(device, commandPool, pAllocator);

    // Remove the command pool from the map if we are taking screenshots.
    loader_platform_thread_lock_mutex(&globalLock);
    if (screenshotFramesReceived && screenshotFrames.empty() && !screenShotFrameRange.valid) {
        // No screenshots in the list to take
        loader_platform_thread_unlock_mutex(&globalLock);
        return;
    }

    // Remove the commandPool from the device mapping
    devMap->commandPools.remove(commandPool);
    loader_platform_thread_unlock_mutex(&globalLock);
    return;
}


VKAPI_ATTR VkResult VKAPI_CALL CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR *pCreateInfo,
                                                  const VkAllocationCallbacks *pAllocator, VkSwapchainKHR *pSwapchain) {
    DeviceMapStruct *devMap = get_dev_info(device);
    assert(devMap);
    VkLayerDispatchTable *pDisp = devMap->device_dispatch_table;

    // This layer does an image copy later on, and the copy command expects the
    // transfer src bit to be on.
    VkSwapchainCreateInfoKHR myCreateInfo = *pCreateInfo;
    myCreateInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkResult result = pDisp->CreateSwapchainKHR(device, &myCreateInfo, pAllocator, pSwapchain);

    // Save the swapchain in a map of we are taking screenshots.
    loader_platform_thread_lock_mutex(&globalLock);
    if (screenshotFramesReceived && screenshotFrames.empty() && !screenShotFrameRange.valid) {
        // No screenshots in the list to take
        loader_platform_thread_unlock_mutex(&globalLock);
        return result;
    }

    if (result == VK_SUCCESS) {
        // Create a mapping for a swapchain to a device, image extent, and
        // format
        SwapchainMapStruct *swapchainMapElem = new SwapchainMapStruct;
        swapchainMapElem->device = device;
        swapchainMapElem->imageExtent = pCreateInfo->imageExtent;
        swapchainMapElem->format = pCreateInfo->imageFormat;
        swapchainMap.insert(make_pair(*pSwapchain, swapchainMapElem));

        // Create a mapping for the swapchain object into the dispatch table
        // TODO is this needed? screenshot_device_table_map.emplace((void
        // *)pSwapchain, pTable);
    }
    loader_platform_thread_unlock_mutex(&globalLock);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain, uint32_t *pCount,
                                                     VkImage *pSwapchainImages) {
    DeviceMapStruct *devMap = get_dev_info(device);
    assert(devMap);
    VkLayerDispatchTable *pDisp = devMap->device_dispatch_table;
    VkResult result = pDisp->GetSwapchainImagesKHR(device, swapchain, pCount, pSwapchainImages);

    // Save the swapchain images in a map if we are taking screenshots
    loader_platform_thread_lock_mutex(&globalLock);
    if (screenshotFramesReceived && screenshotFrames.empty() && !screenShotFrameRange.valid) {
        // No screenshots in the list to take
        loader_platform_thread_unlock_mutex(&globalLock);
        return result;
    }

    if (result == VK_SUCCESS && pSwapchainImages && !swapchainMap.empty() && swapchainMap.find(swapchain) != swapchainMap.end()) {
        unsigned i;

        for (i = 0; i < *pCount; i++) {
            // Create a mapping for an image to a device, image extent, and
            // format
            if (imageMap[pSwapchainImages[i]] == NULL) {
                ImageMapStruct *imageMapElem = new ImageMapStruct;
                imageMap[pSwapchainImages[i]] = imageMapElem;
            }
            imageMap[pSwapchainImages[i]]->device = swapchainMap[swapchain]->device;
            imageMap[pSwapchainImages[i]]->imageExtent = swapchainMap[swapchain]->imageExtent;
            imageMap[pSwapchainImages[i]]->format = swapchainMap[swapchain]->format;
        }

        // Add list of images to swapchain to image map
        SwapchainMapStruct *swapchainMapElem = swapchainMap[swapchain];
        if (i >= 1 && swapchainMapElem) {
            VkImage *imageList = new VkImage[i];
            swapchainMapElem->imageList = imageList;
            for (unsigned j = 0; j < i; j++) {
                swapchainMapElem->imageList[j] = pSwapchainImages[j];
            }
        }
    }
    loader_platform_thread_unlock_mutex(&globalLock);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pPresentInfo) {
    static int frameNumber = 0;
    if (frameNumber == 10) {
        fflush(stdout); /* *((int*)0)=0; */
    }
    DeviceMapStruct *devMap = get_dev_info((VkDevice)queue);
    assert(devMap);
    VkLayerDispatchTable *pDisp = devMap->device_dispatch_table;
    VkResult result = pDisp->QueuePresentKHR(queue, pPresentInfo);
    loader_platform_thread_lock_mutex(&globalLock);

    if (!screenshotFramesReceived) {
        const char *vk_screenshot_frames = local_getenv(env_var);
        if (vk_screenshot_frames && *vk_screenshot_frames) {
            populate_frame_list(vk_screenshot_frames);
        }
        // Backwards compatibility
        else {
            const char *_vk_screenshot = local_getenv(env_var_old);
            if (_vk_screenshot && *_vk_screenshot) {
                populate_frame_list(_vk_screenshot);
            }
            local_free_getenv(_vk_screenshot);
        }

        local_free_getenv(vk_screenshot_frames);
    }

    if (result == VK_SUCCESS && (!screenshotFrames.empty() || screenShotFrameRange.valid)) {
        set<int>::iterator it;
        bool inScreenShotFrames = false;
        bool inScreenShotFrameRange = false;
        it = screenshotFrames.find(frameNumber);
        inScreenShotFrames = (it != screenshotFrames.end());
        isInScreenShotFrameRange(frameNumber, &screenShotFrameRange, &inScreenShotFrameRange);
        if ((inScreenShotFrames) || (inScreenShotFrameRange)) {
            string fileName;

#ifdef ANDROID
            // std::to_string is not supported currently
            char buffer[64];
            snprintf(buffer, sizeof(buffer), "/sdcard/Android/%d", frameNumber);
            std::string base(buffer);
            fileName = base + ".ppm";
#else
            fileName = to_string(frameNumber) + ".ppm";
            printf("Screen Capture file is: %s \n", fileName.c_str());
#endif

            VkImage image;
            VkSwapchainKHR swapchain;
            // We'll dump only one image: the first
            swapchain = pPresentInfo->pSwapchains[0];
            image = swapchainMap[swapchain]->imageList[pPresentInfo->pImageIndices[0]];
            writePPM(fileName.c_str(), image);
            if (inScreenShotFrames) {
                screenshotFrames.erase(it);
            }

            if (screenshotFrames.empty() && isEndOfScreenShotFrameRange(frameNumber, &screenShotFrameRange)) {
                // Free all our maps since we are done with them.
                for (auto swapchainIter = swapchainMap.begin(); swapchainIter != swapchainMap.end(); swapchainIter++) {
                    SwapchainMapStruct *swapchainMapElem = swapchainIter->second;
                    delete swapchainMapElem;
                }
                for (auto imageIter = imageMap.begin(); imageIter != imageMap.end(); imageIter++) {
                    ImageMapStruct *imageMapElem = imageIter->second;
                    delete imageMapElem;
                }
                for (auto physDeviceIter = physDeviceMap.begin(); physDeviceIter != physDeviceMap.end(); physDeviceIter++) {
                    PhysDeviceMapStruct *physDeviceMapElem = physDeviceIter->second;
                    delete physDeviceMapElem;
                }
                swapchainMap.clear();
                imageMap.clear();
                physDeviceMap.clear();
                screenShotFrameRange.valid = false;
            }
        }
    }
    frameNumber++;
    loader_platform_thread_unlock_mutex(&globalLock);
    return result;
}

// Unused, but this could be provided as an extension or utility to the
// application in the future.
VKAPI_ATTR VkResult VKAPI_CALL SpecifyScreenshotFrames(const char *frameList) {
    populate_frame_list(frameList);
    return VK_SUCCESS;
}

static const VkLayerProperties global_layer = {
    "VK_LAYER_LUNARG_screenshot", VK_MAKE_VERSION(1, 0, VK_HEADER_VERSION), 1, "Layer: screenshot",
};

VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceLayerProperties(uint32_t *pCount, VkLayerProperties *pProperties) {
    return util_GetLayerProperties(1, &global_layer, pCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pCount,
                                                              VkLayerProperties *pProperties) {
    return util_GetLayerProperties(1, &global_layer, pCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pCount,
                                                                    VkExtensionProperties *pProperties) {
    if (pLayerName && !strcmp(pLayerName, global_layer.layerName)) return util_GetExtensionProperties(0, NULL, pCount, pProperties);

    return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *pLayerName,
                                                                  uint32_t *pCount, VkExtensionProperties *pProperties) {
    if (pLayerName && !strcmp(pLayerName, global_layer.layerName)) return util_GetExtensionProperties(0, NULL, pCount, pProperties);

    assert(physicalDevice);

    VkLayerInstanceDispatchTable *pTable = instance_dispatch_table(physicalDevice);
    return pTable->EnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pCount, pProperties);
}

static PFN_vkVoidFunction intercept_core_instance_command(const char *name);

static PFN_vkVoidFunction intercept_core_device_command(const char *name);

static PFN_vkVoidFunction intercept_khr_swapchain_command(const char *name, VkDevice dev);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice dev, const char *funcName) {
    PFN_vkVoidFunction proc = intercept_core_device_command(funcName);
    if (proc) return proc;

    if (dev == NULL) {
        return NULL;
    }

    proc = intercept_khr_swapchain_command(funcName, dev);
    if (proc) return proc;

    DeviceMapStruct *devMap = get_dev_info(dev);
    assert(devMap);
    VkLayerDispatchTable *pDisp = devMap->device_dispatch_table;

    if (pDisp->GetDeviceProcAddr == NULL) return NULL;
    return pDisp->GetDeviceProcAddr(dev, funcName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance instance, const char *funcName) {
    PFN_vkVoidFunction proc = intercept_core_instance_command(funcName);
    if (proc) return proc;

    assert(instance);

    proc = intercept_core_device_command(funcName);
    if (!proc) proc = intercept_khr_swapchain_command(funcName, VK_NULL_HANDLE);
    if (proc) return proc;

    VkLayerInstanceDispatchTable *pTable = instance_dispatch_table(instance);
    if (pTable->GetInstanceProcAddr == NULL) return NULL;
    return pTable->GetInstanceProcAddr(instance, funcName);
}

static PFN_vkVoidFunction intercept_core_instance_command(const char *name) {
    static const struct {
        const char *name;
        PFN_vkVoidFunction proc;
    } core_instance_commands[] = {
        {"vkGetInstanceProcAddr", reinterpret_cast<PFN_vkVoidFunction>(GetInstanceProcAddr)},
        {"vkCreateInstance", reinterpret_cast<PFN_vkVoidFunction>(CreateInstance)},
        {"vkCreateDevice", reinterpret_cast<PFN_vkVoidFunction>(CreateDevice)},
        {"vkEnumeratePhysicalDevices", reinterpret_cast<PFN_vkVoidFunction>(EnumeratePhysicalDevices)},
        {"vkEnumerateInstanceLayerProperties", reinterpret_cast<PFN_vkVoidFunction>(EnumerateInstanceLayerProperties)},
        {"vkEnumerateDeviceLayerProperties", reinterpret_cast<PFN_vkVoidFunction>(EnumerateDeviceLayerProperties)},
        {"vkEnumerateInstanceExtensionProperties", reinterpret_cast<PFN_vkVoidFunction>(EnumerateInstanceExtensionProperties)},
        {"vkEnumerateDeviceExtensionProperties", reinterpret_cast<PFN_vkVoidFunction>(EnumerateDeviceExtensionProperties)}};

    for (size_t i = 0; i < ARRAY_SIZE(core_instance_commands); i++) {
        if (!strcmp(core_instance_commands[i].name, name)) return core_instance_commands[i].proc;
    }

    return nullptr;
}

static PFN_vkVoidFunction intercept_core_device_command(const char *name) {
    static const struct {
        const char *name;
        PFN_vkVoidFunction proc;
    } core_device_commands[] = {
        {"vkGetDeviceProcAddr", reinterpret_cast<PFN_vkVoidFunction>(GetDeviceProcAddr)},
        {"vkGetDeviceQueue", reinterpret_cast<PFN_vkVoidFunction>(GetDeviceQueue)},
        {"vkCreateCommandPool", reinterpret_cast<PFN_vkVoidFunction>(CreateCommandPool)},
        {"vkDestroyCommandPool", reinterpret_cast<PFN_vkVoidFunction>(DestroyCommandPool)},
        {"vkDestroyDevice", reinterpret_cast<PFN_vkVoidFunction>(DestroyDevice)},
    };

    for (size_t i = 0; i < ARRAY_SIZE(core_device_commands); i++) {
        if (!strcmp(core_device_commands[i].name, name)) return core_device_commands[i].proc;
    }

    return nullptr;
}

static PFN_vkVoidFunction intercept_khr_swapchain_command(const char *name, VkDevice dev) {
    static const struct {
        const char *name;
        PFN_vkVoidFunction proc;
    } khr_swapchain_commands[] = {
        {"vkCreateSwapchainKHR", reinterpret_cast<PFN_vkVoidFunction>(CreateSwapchainKHR)},
        {"vkGetSwapchainImagesKHR", reinterpret_cast<PFN_vkVoidFunction>(GetSwapchainImagesKHR)},
        {"vkQueuePresentKHR", reinterpret_cast<PFN_vkVoidFunction>(QueuePresentKHR)},
    };

    if (dev) {
        DeviceMapStruct *devMap = get_dev_info(dev);
        if (!devMap->wsi_enabled) return nullptr;
    }

    for (size_t i = 0; i < ARRAY_SIZE(khr_swapchain_commands); i++) {
        if (!strcmp(khr_swapchain_commands[i].name, name)) return khr_swapchain_commands[i].proc;
    }

    return nullptr;
}

}  // namespace screenshot

// loader-layer interface v0, just wrappers since there is only a layer

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pCount,
                                                                                  VkLayerProperties *pProperties) {
    return screenshot::EnumerateInstanceLayerProperties(pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pCount,
                                                                                VkLayerProperties *pProperties) {
    // the layer command handles VK_NULL_HANDLE just fine internally
    assert(physicalDevice == VK_NULL_HANDLE);
    return screenshot::EnumerateDeviceLayerProperties(VK_NULL_HANDLE, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pCount,
                                                                                      VkExtensionProperties *pProperties) {
    return screenshot::EnumerateInstanceExtensionProperties(pLayerName, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                                                    const char *pLayerName, uint32_t *pCount,
                                                                                    VkExtensionProperties *pProperties) {
    // the layer command handles VK_NULL_HANDLE just fine internally
    assert(physicalDevice == VK_NULL_HANDLE);
    return screenshot::EnumerateDeviceExtensionProperties(VK_NULL_HANDLE, pLayerName, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice dev, const char *funcName) {
    return screenshot::GetDeviceProcAddr(dev, funcName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *funcName) {
    return screenshot::GetInstanceProcAddr(instance, funcName);
}
