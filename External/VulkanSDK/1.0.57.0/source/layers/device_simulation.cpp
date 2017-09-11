/*
 * Copyright (C) 2015-2017 Valve Corporation
 * Copyright (C) 2015-2017 LunarG, Inc.
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
 * Author: Mike Weiblen <mikew@lunarg.com>
 * Author: Arda Coskunses <arda@lunarg.com>
 */

/*
 * layersvt/device_simulation.cpp - The VK_LAYER_LUNARG_device_simulation layer.
 * This DevSim layer simulates a device by loading a JSON configuration file to override values that would normally be returned
 * from a Vulkan implementation.  The configuration files must validate with the DevSim schema; see the kSchemaDevsim100 variable
 * below for the schema's URI.
 *
 * References (several documents are also included in the LunarG Vulkan SDK, see [SDK]):
 * [SPEC]   https://www.khronos.org/registry/vulkan/specs/1.0-extensions/html/vkspec.html
 * [SDK]    https://vulkan.lunarg.com/sdk/home
 * [LALI]   https://github.com/KhronosGroup/Vulkan-LoaderAndValidationLayers/blob/master/loader/LoaderAndLayerInterface.md
 *
 * Misc notes:
 * This code generally follows the spirit of the Google C++ styleguide, while accommodating conventions of the Vulkan styleguide.
 * https://google.github.io/styleguide/cppguide.html
 * https://www.khronos.org/registry/vulkan/specs/1.0/styleguide.html
 */

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <functional>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <mutex>

#include <json/json.h>  // https://github.com/open-source-parsers/jsoncpp

#include "vulkan/vk_layer.h"
#include "vk_layer_table.h"

namespace {

// Global constants //////////////////////////////////////////////////////////////////////////////////////////////////////////////

const uint32_t kVersionDevsimMajor = 1;
const uint32_t kVersionDevsimMinor = 0;
const uint32_t kVersionDevsimPatch = 0;
const uint32_t kVersionDevsimImplementation = VK_MAKE_VERSION(kVersionDevsimMajor, kVersionDevsimMinor, kVersionDevsimPatch);

const VkLayerProperties kLayerProperties[] = {{
    "VK_LAYER_LUNARG_device_simulation",       // layerName
    VK_MAKE_VERSION(1, 0, VK_HEADER_VERSION),  // specVersion
    kVersionDevsimImplementation,              // implementationVersion
    "LunarG device simulation layer"           // description
}};
const uint32_t kLayerPropertiesCount = (sizeof(kLayerProperties) / sizeof(kLayerProperties[0]));

const VkExtensionProperties *kExtensionProperties = nullptr;
const uint32_t kExtensionPropertiesCount = 0;

const char *kSchemaDevsim100 = "https://schema.khronos.org/vulkan/devsim_1_0_0.json#";

// Environment variables defined by this layer ///////////////////////////////////////////////////////////////////////////////////

const char *const kEnvarDevsimFilename = "VK_DEVSIM_FILENAME";          // name of the configuration file to load.
const char *const kEnvarDevsimDebugEnable = "VK_DEVSIM_DEBUG_ENABLE";   // a non-zero integer will enable debugging output.
const char *const kEnvarDevsimExitOnError = "VK_DEVSIM_EXIT_ON_ERROR";  // a non-zero integer will enable exit-on-error.

// Various small utility functions ///////////////////////////////////////////////////////////////////////////////////////////////

// Retrieve the value of an environment variable.
std::string GetEnvarValue(const char *name) {
    std::string value = "";
#if defined(_WIN32)
    DWORD size = GetEnvironmentVariable(name, nullptr, 0);
    if (size > 0) {
        std::vector<char> buffer(size);
        GetEnvironmentVariable(name, buffer.data(), size);
        value = buffer.data();
    }
#elif defined(__ANDROID__)
#error "TODO Android not implemented yet"
#else
    const char *v = getenv(name);
    if (v) value = v;
#endif
    // printf("envar %s = \"%s\"\n", name, value.c_str());
    return value;
}

void DebugPrintf(const char *fmt, ...) {
    static const int kDebugLevel = std::atoi(GetEnvarValue(kEnvarDevsimDebugEnable).c_str());
    if (kDebugLevel > 0) {
        printf("\tDEBUG devsim ");
        va_list args;
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
    }
}

void ErrorPrintf(const char *fmt, ...) {
    static const int kExitLevel = std::atoi(GetEnvarValue(kEnvarDevsimExitOnError).c_str());
    fprintf(stderr, "\tERROR devsim ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    if (kExitLevel > 0) {
        fprintf(stderr, "\ndevsim exiting on error as requested\n\n");
        exit(1);
    }
}

// Get all elements from a vkEnumerate*() lambda into a properly-sized std::vector.
template <typename T>
VkResult EnumerateAll(std::vector<T> *vect, std::function<VkResult(uint32_t *, T *)> func) {
    VkResult result = VK_INCOMPLETE;
    do {
        uint32_t count = 0;
        result = func(&count, nullptr);
        assert(!result);
        vect->resize(count);
        result = func(&count, vect->data());
    } while (result == VK_INCOMPLETE);
    return result;
}

// Global variables //////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::mutex global_lock;  // Enforce thread-safety for this layer's containers.

uint32_t loader_layer_iface_version = CURRENT_LOADER_LAYER_INTERFACE_VERSION;

// PhysicalDeviceData : creates and manages the simulated device configurations //////////////////////////////////////////////////

class PhysicalDeviceData {
   public:
    // Create a new PDD element, allocated from our map.
    static PhysicalDeviceData &Create(VkPhysicalDevice pd, VkInstance instance) {
        assert(!Find(pd));  // Verify this instance does not already exist.
        const auto result = map_.emplace(pd, PhysicalDeviceData(pd, instance));
        assert(result.second);  // true=insertion, false=replacement
        auto iter = result.first;
        PhysicalDeviceData *pdd = &iter->second;
        assert(Find(pd) == pdd);  // Verify we get the same instance we just inserted.
        DebugPrintf("PDD Create() physical_device %p pdd %p\n", pd, pdd);
        return *pdd;
    }

    // Find a PDD from our map, or nullptr if doesn't exist.
    static PhysicalDeviceData *Find(VkPhysicalDevice pd) {
        const auto iter = map_.find(pd);
        return (iter != map_.end()) ? &iter->second : nullptr;
    }

    VkInstance instance() const { return instance_; }

    VkPhysicalDeviceProperties physical_device_properties_;
    VkPhysicalDeviceFeatures physical_device_features_;

   private:
    PhysicalDeviceData() = delete;
    PhysicalDeviceData &operator=(const PhysicalDeviceData &) = delete;
    PhysicalDeviceData(VkPhysicalDevice pd, VkInstance instance) : physical_device_(pd), instance_(instance) {}

    const VkPhysicalDevice physical_device_;
    const VkInstance instance_;

    typedef std::unordered_map<VkPhysicalDevice, PhysicalDeviceData> Map;
    static Map map_;
};

PhysicalDeviceData::Map PhysicalDeviceData::map_;

// Loader for DevSim JSON configuration files ////////////////////////////////////////////////////////////////////////////////////

class JsonLoader {
   public:
    JsonLoader(PhysicalDeviceData &pdd) : pdd_(pdd) {}
    JsonLoader() = delete;
    JsonLoader(const JsonLoader &) = delete;
    JsonLoader &operator=(const JsonLoader &) = delete;

    bool LoadFile(const char *filename);

   private:
    void ApplyOverrides(const Json::Value &value, VkPhysicalDeviceProperties *dest);
    void ApplyOverrides(const Json::Value &value, VkPhysicalDeviceLimits *dest);
    void ApplyOverrides(const Json::Value &value, VkPhysicalDeviceSparseProperties *dest);
    void ApplyOverrides(const Json::Value &value, VkPhysicalDeviceFeatures *dest);

    void GetValue(const Json::Value &value, float *dest) {
        if (!value.isNull()) {
            *dest = value.asFloat();
        }
    }

    void GetValue(const Json::Value &value, int32_t *dest) {
        if (!value.isNull()) {
            *dest = value.asInt();
        }
    }

    void GetValue(const Json::Value &value, uint32_t *dest) {
        if (value.isBool()) {  // for VkBool32
            *dest = value.asBool() ? VK_TRUE : VK_FALSE;
        } else if (!value.isNull()) {
            *dest = value.asUInt();
        }
    }

    void GetValue(const Json::Value &value, uint64_t *dest) {
        if (!value.isNull()) {
            *dest = value.asUInt64();
        }
    }

    template <typename T>  // for Vulkan enum types
    void GetValue(const Json::Value &value, T *dest) {
        if (!value.isNull()) {
            *dest = static_cast<T>(value.asInt());
        }
    }

    void GetArray(const Json::Value &value, int count, uint8_t *dest) {
        if (!value.isNull()) {
            for (int i = 0; i < count; ++i) {
                dest[i] = value[i].asUInt();
            }
        }
    }

    void GetArray(const Json::Value &value, int count, uint32_t *dest) {
        if (!value.isNull()) {
            for (int i = 0; i < count; ++i) {
                dest[i] = value[i].asUInt();
            }
        }
    }

    void GetArray(const Json::Value &value, int count, float *dest) {
        if (!value.isNull()) {
            for (int i = 0; i < count; ++i) {
                dest[i] = value[i].asFloat();
            }
        }
    }

    void GetArray(const Json::Value &value, int count, char *dest) {
        if (!value.isNull()) {
            dest[0] = '\0';
            strncpy(dest, value.asCString(), count);
            dest[count - 1] = '\0';
        }
    }

    PhysicalDeviceData &pdd_;
};

bool JsonLoader::LoadFile(const char *filename) {
    std::ifstream json_file(filename);
    if (!json_file) {
        ErrorPrintf("JsonLoader failed to open file \"%s\"\n", filename);
        return false;
    }

    DebugPrintf("JsonCpp version %s\n", JSONCPP_VERSION_STRING);
    Json::Reader reader;
    Json::Value root = Json::nullValue;
    bool success = reader.parse(json_file, root, false);
    if (!success) {
        ErrorPrintf("Json::Reader failed {\n%s}\n", reader.getFormattedErrorMessages().c_str());
        return false;
    }
    json_file.close();

    if (!root.isObject()) {
        ErrorPrintf("Json document root is not an object\n");
        return false;
    }
    DebugPrintf("\t\tJsonLoader::LoadFile() OK\n");

    ApplyOverrides(root["VkPhysicalDeviceProperties"], &pdd_.physical_device_properties_);
    ApplyOverrides(root["VkPhysicalDeviceFeatures"], &pdd_.physical_device_features_);

    return true;
}

// Apply the DRY principle, see https://en.wikipedia.org/wiki/Don%27t_repeat_yourself
#define GET_VALUE(x) GetValue(value[#x], &dest->x)
#define GET_ARRAY(x, n) GetArray(value[#x], n, dest->x)

void JsonLoader::ApplyOverrides(const Json::Value &value, VkPhysicalDeviceProperties *dest) {
    DebugPrintf("\t\tJsonLoader::ApplyOverrides() VkPhysicalDeviceProperties\n");
    if (value.isNull()) {
        return;
    } else if (!value.isObject()) {
        ErrorPrintf("JSON element \"VkPhysicalDeviceProperties\" is not an object\n");
        return;
    }

    GET_VALUE(apiVersion);
    GET_VALUE(driverVersion);
    GET_VALUE(vendorID);
    GET_VALUE(deviceID);
    GET_VALUE(deviceType);
    GET_ARRAY(deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE);
    GET_ARRAY(pipelineCacheUUID, VK_UUID_SIZE);
    ApplyOverrides(value["limits"], &dest->limits);
    ApplyOverrides(value["sparseProperties"], &dest->sparseProperties);
}

void JsonLoader::ApplyOverrides(const Json::Value &value, VkPhysicalDeviceLimits *dest) {
    DebugPrintf("\t\tJsonLoader::ApplyOverrides() VkPhysicalDeviceLimits\n");
    if (value.isNull()) {
        return;
    } else if (!value.isObject()) {
        ErrorPrintf("JSON element \"limits\" is not an object\n");
        return;
    }

    GET_VALUE(maxImageDimension1D);
    GET_VALUE(maxImageDimension2D);
    GET_VALUE(maxImageDimension3D);
    GET_VALUE(maxImageDimensionCube);
    GET_VALUE(maxImageArrayLayers);
    GET_VALUE(maxTexelBufferElements);
    GET_VALUE(maxUniformBufferRange);
    GET_VALUE(maxStorageBufferRange);
    GET_VALUE(maxPushConstantsSize);
    GET_VALUE(maxMemoryAllocationCount);
    GET_VALUE(maxSamplerAllocationCount);
    GET_VALUE(bufferImageGranularity);
    GET_VALUE(sparseAddressSpaceSize);
    GET_VALUE(maxBoundDescriptorSets);
    GET_VALUE(maxPerStageDescriptorSamplers);
    GET_VALUE(maxPerStageDescriptorUniformBuffers);
    GET_VALUE(maxPerStageDescriptorStorageBuffers);
    GET_VALUE(maxPerStageDescriptorSampledImages);
    GET_VALUE(maxPerStageDescriptorStorageImages);
    GET_VALUE(maxPerStageDescriptorInputAttachments);
    GET_VALUE(maxPerStageResources);
    GET_VALUE(maxDescriptorSetSamplers);
    GET_VALUE(maxDescriptorSetUniformBuffers);
    GET_VALUE(maxDescriptorSetUniformBuffersDynamic);
    GET_VALUE(maxDescriptorSetStorageBuffers);
    GET_VALUE(maxDescriptorSetStorageBuffersDynamic);
    GET_VALUE(maxDescriptorSetSampledImages);
    GET_VALUE(maxDescriptorSetStorageImages);
    GET_VALUE(maxDescriptorSetInputAttachments);
    GET_VALUE(maxVertexInputAttributes);
    GET_VALUE(maxVertexInputBindings);
    GET_VALUE(maxVertexInputAttributeOffset);
    GET_VALUE(maxVertexInputBindingStride);
    GET_VALUE(maxVertexOutputComponents);
    GET_VALUE(maxTessellationGenerationLevel);
    GET_VALUE(maxTessellationPatchSize);
    GET_VALUE(maxTessellationControlPerVertexInputComponents);
    GET_VALUE(maxTessellationControlPerVertexOutputComponents);
    GET_VALUE(maxTessellationControlPerPatchOutputComponents);
    GET_VALUE(maxTessellationControlTotalOutputComponents);
    GET_VALUE(maxTessellationEvaluationInputComponents);
    GET_VALUE(maxTessellationEvaluationOutputComponents);
    GET_VALUE(maxGeometryShaderInvocations);
    GET_VALUE(maxGeometryInputComponents);
    GET_VALUE(maxGeometryOutputComponents);
    GET_VALUE(maxGeometryOutputVertices);
    GET_VALUE(maxGeometryTotalOutputComponents);
    GET_VALUE(maxFragmentInputComponents);
    GET_VALUE(maxFragmentOutputAttachments);
    GET_VALUE(maxFragmentDualSrcAttachments);
    GET_VALUE(maxFragmentCombinedOutputResources);
    GET_VALUE(maxComputeSharedMemorySize);
    GET_ARRAY(maxComputeWorkGroupCount, 3);
    GET_VALUE(maxComputeWorkGroupInvocations);
    GET_ARRAY(maxComputeWorkGroupSize, 3);
    GET_VALUE(subPixelPrecisionBits);
    GET_VALUE(subTexelPrecisionBits);
    GET_VALUE(mipmapPrecisionBits);
    GET_VALUE(maxDrawIndexedIndexValue);
    GET_VALUE(maxDrawIndirectCount);
    GET_VALUE(maxSamplerLodBias);
    GET_VALUE(maxSamplerAnisotropy);
    GET_VALUE(maxViewports);
    GET_ARRAY(maxViewportDimensions, 2);
    GET_ARRAY(viewportBoundsRange, 2);
    GET_VALUE(viewportSubPixelBits);
    GET_VALUE(minMemoryMapAlignment);
    GET_VALUE(minTexelBufferOffsetAlignment);
    GET_VALUE(minUniformBufferOffsetAlignment);
    GET_VALUE(minStorageBufferOffsetAlignment);
    GET_VALUE(minTexelOffset);
    GET_VALUE(maxTexelOffset);
    GET_VALUE(minTexelGatherOffset);
    GET_VALUE(maxTexelGatherOffset);
    GET_VALUE(minInterpolationOffset);
    GET_VALUE(maxInterpolationOffset);
    GET_VALUE(subPixelInterpolationOffsetBits);
    GET_VALUE(maxFramebufferWidth);
    GET_VALUE(maxFramebufferHeight);
    GET_VALUE(maxFramebufferLayers);
    GET_VALUE(framebufferColorSampleCounts);
    GET_VALUE(framebufferDepthSampleCounts);
    GET_VALUE(framebufferStencilSampleCounts);
    GET_VALUE(framebufferNoAttachmentsSampleCounts);
    GET_VALUE(maxColorAttachments);
    GET_VALUE(sampledImageColorSampleCounts);
    GET_VALUE(sampledImageIntegerSampleCounts);
    GET_VALUE(sampledImageDepthSampleCounts);
    GET_VALUE(sampledImageStencilSampleCounts);
    GET_VALUE(storageImageSampleCounts);
    GET_VALUE(maxSampleMaskWords);
    GET_VALUE(timestampComputeAndGraphics);
    GET_VALUE(timestampPeriod);
    GET_VALUE(maxClipDistances);
    GET_VALUE(maxCullDistances);
    GET_VALUE(maxCombinedClipAndCullDistances);
    GET_VALUE(discreteQueuePriorities);
    GET_ARRAY(pointSizeRange, 2);
    GET_ARRAY(lineWidthRange, 2);
    GET_VALUE(pointSizeGranularity);
    GET_VALUE(lineWidthGranularity);
    GET_VALUE(strictLines);
    GET_VALUE(standardSampleLocations);
    GET_VALUE(optimalBufferCopyOffsetAlignment);
    GET_VALUE(optimalBufferCopyRowPitchAlignment);
    GET_VALUE(nonCoherentAtomSize);
}

void JsonLoader::ApplyOverrides(const Json::Value &value, VkPhysicalDeviceSparseProperties *dest) {
    DebugPrintf("\t\tJsonLoader::ApplyOverrides() VkPhysicalDeviceSparseProperties\n");
    if (value.isNull()) {
        return;
    } else if (!value.isObject()) {
        ErrorPrintf("JSON element \"sparseProperties\" is not an object\n");
        return;
    }

    GET_VALUE(residencyStandard2DBlockShape);
    GET_VALUE(residencyStandard2DMultisampleBlockShape);
    GET_VALUE(residencyStandard3DBlockShape);
    GET_VALUE(residencyAlignedMipSize);
    GET_VALUE(residencyNonResidentStrict);
}

void JsonLoader::ApplyOverrides(const Json::Value &value, VkPhysicalDeviceFeatures *dest) {
    DebugPrintf("\t\tJsonLoader::ApplyOverrides() VkPhysicalDeviceFeatures\n");
    if (value.isNull()) {
        return;
    } else if (!value.isObject()) {
        ErrorPrintf("JSON element \"VkPhysicalDeviceFeatures\" is not an object\n");
        return;
    }

    GET_VALUE(robustBufferAccess);
    GET_VALUE(fullDrawIndexUint32);
    GET_VALUE(imageCubeArray);
    GET_VALUE(independentBlend);
    GET_VALUE(geometryShader);
    GET_VALUE(tessellationShader);
    GET_VALUE(sampleRateShading);
    GET_VALUE(dualSrcBlend);
    GET_VALUE(logicOp);
    GET_VALUE(multiDrawIndirect);
    GET_VALUE(drawIndirectFirstInstance);
    GET_VALUE(depthClamp);
    GET_VALUE(depthBiasClamp);
    GET_VALUE(fillModeNonSolid);
    GET_VALUE(depthBounds);
    GET_VALUE(wideLines);
    GET_VALUE(largePoints);
    GET_VALUE(alphaToOne);
    GET_VALUE(multiViewport);
    GET_VALUE(samplerAnisotropy);
    GET_VALUE(textureCompressionETC2);
    GET_VALUE(textureCompressionASTC_LDR);
    GET_VALUE(textureCompressionBC);
    GET_VALUE(occlusionQueryPrecise);
    GET_VALUE(pipelineStatisticsQuery);
    GET_VALUE(vertexPipelineStoresAndAtomics);
    GET_VALUE(fragmentStoresAndAtomics);
    GET_VALUE(shaderTessellationAndGeometryPointSize);
    GET_VALUE(shaderImageGatherExtended);
    GET_VALUE(shaderStorageImageExtendedFormats);
    GET_VALUE(shaderStorageImageMultisample);
    GET_VALUE(shaderStorageImageReadWithoutFormat);
    GET_VALUE(shaderStorageImageWriteWithoutFormat);
    GET_VALUE(shaderUniformBufferArrayDynamicIndexing);
    GET_VALUE(shaderSampledImageArrayDynamicIndexing);
    GET_VALUE(shaderStorageBufferArrayDynamicIndexing);
    GET_VALUE(shaderStorageImageArrayDynamicIndexing);
    GET_VALUE(shaderClipDistance);
    GET_VALUE(shaderCullDistance);
    GET_VALUE(shaderFloat64);
    GET_VALUE(shaderInt64);
    GET_VALUE(shaderInt16);
    GET_VALUE(shaderResourceResidency);
    GET_VALUE(shaderResourceMinLod);
    GET_VALUE(sparseBinding);
    GET_VALUE(sparseResidencyBuffer);
    GET_VALUE(sparseResidencyImage2D);
    GET_VALUE(sparseResidencyImage3D);
    GET_VALUE(sparseResidency2Samples);
    GET_VALUE(sparseResidency4Samples);
    GET_VALUE(sparseResidency8Samples);
    GET_VALUE(sparseResidency16Samples);
    GET_VALUE(sparseResidencyAliased);
    GET_VALUE(variableMultisampleRate);
    GET_VALUE(inheritedQueries);
}

#undef GET_VALUE
#undef GET_ARRAY

// Layer-specific wrappers for Vulkan functions, accessed via vkGet*ProcAddr() ///////////////////////////////////////////////////

// Generic layer dispatch table setup, see [LALI].
VkResult LayerSetupCreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                                  VkInstance *pInstance) {
    VkLayerInstanceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    assert(chain_info->u.pLayerInfo);

    PFN_vkGetInstanceProcAddr fp_get_instance_proc_addr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateInstance fp_create_instance = (PFN_vkCreateInstance)fp_get_instance_proc_addr(nullptr, "vkCreateInstance");
    if (!fp_create_instance) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;
    VkResult result = fp_create_instance(pCreateInfo, pAllocator, pInstance);
    if (!result) {
        initInstanceTable(*pInstance, fp_get_instance_proc_addr);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                                              VkInstance *pInstance) {
    DebugPrintf("CreateInstance START {\n");

    std::lock_guard<std::mutex> lock(global_lock);

    VkResult result = LayerSetupCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (result) {
        return result;
    }

    // Our layer-specific initialization...

    DebugPrintf("%s version %d.%d.%d\n", kLayerProperties[0].layerName, kVersionDevsimMajor, kVersionDevsimMinor,
                kVersionDevsimPatch);

    // Get the name of our configuration file.
    std::string filename = GetEnvarValue(kEnvarDevsimFilename);
    DebugPrintf("\t\tenvar %s = \"%s\"\n", kEnvarDevsimFilename, filename.c_str());
    if (filename.empty()) {
        ErrorPrintf("envar %s is unset\n", kEnvarDevsimFilename);
    }

    const auto dt = instance_dispatch_table(*pInstance);

    std::vector<VkPhysicalDevice> physical_devices;
    result = EnumerateAll<VkPhysicalDevice>(&physical_devices, [&](uint32_t *count, VkPhysicalDevice *values) {
        return dt->EnumeratePhysicalDevices(*pInstance, count, values);
    });
    if (result) {
        return result;
    }

    // For each physical device, create and populate a PDD instance.
    for (const auto &physical_device : physical_devices) {
        PhysicalDeviceData &pdd = PhysicalDeviceData::Create(physical_device, *pInstance);

        // Initialize PDD to the actual Vulkan implementation's defaults.
        dt->GetPhysicalDeviceProperties(physical_device, &pdd.physical_device_properties_);
        dt->GetPhysicalDeviceFeatures(physical_device, &pdd.physical_device_features_);

        // Apply override values from the configuration file.
        JsonLoader json_loader(pdd);
        json_loader.LoadFile(filename.c_str());
    }

    DebugPrintf("CreateInstance END instance %p }\n", *pInstance);
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    DebugPrintf("DestroyInstance instance %p\n", instance);

    std::lock_guard<std::mutex> lock(global_lock);

    {
        const auto dt = instance_dispatch_table(instance);
        dt->DestroyInstance(instance, pAllocator);
    }
    destroy_instance_dispatch_table(get_dispatch_key(instance));
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties *pProperties) {
    std::lock_guard<std::mutex> lock(global_lock);
    const auto dt = instance_dispatch_table(physicalDevice);

    PhysicalDeviceData *pdd = PhysicalDeviceData::Find(physicalDevice);
    DebugPrintf("GetPhysicalDeviceProperties physicalDevice %p pdd %p\n", physicalDevice, pdd);
    if (pdd) {
        *pProperties = pdd->physical_device_properties_;
    } else {
        dt->GetPhysicalDeviceProperties(physicalDevice, pProperties);
    }
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFeatures(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures *pFeatures) {
    std::lock_guard<std::mutex> lock(global_lock);
    const auto dt = instance_dispatch_table(physicalDevice);

    PhysicalDeviceData *pdd = PhysicalDeviceData::Find(physicalDevice);
    DebugPrintf("GetPhysicalDeviceFeatures physicalDevice %p pdd %p\n", physicalDevice, pdd);
    if (pdd) {
        *pFeatures = pdd->physical_device_features_;
    } else {
        dt->GetPhysicalDeviceFeatures(physicalDevice, pFeatures);
    }
}

template <typename T>
VkResult EnumerateProperties(uint32_t src_count, const T *src_props, uint32_t *dst_count, T *dst_props) {
    assert(dst_count);
    if (!dst_props || !src_props) {
        *dst_count = src_count;
        return VK_SUCCESS;
    }

    uint32_t copy_count = (*dst_count < src_count) ? *dst_count : src_count;
    memcpy(dst_props, src_props, copy_count * sizeof(T));
    *dst_count = copy_count;
    return (copy_count == src_count) ? VK_SUCCESS : VK_INCOMPLETE;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceLayerProperties(uint32_t *pCount, VkLayerProperties *pProperties) {
    DebugPrintf("EnumerateInstanceLayerProperties\n");
    return EnumerateProperties(kLayerPropertiesCount, kLayerProperties, pCount, pProperties);
}

// Per [LALI], EnumerateDeviceLayerProperties() is deprecated and may be omitted.

VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pCount,
                                                                    VkExtensionProperties *pProperties) {
    DebugPrintf("EnumerateInstanceExtensionProperties pLayerName \"%s\"\n", pLayerName);
    if (pLayerName && !strcmp(pLayerName, kLayerProperties->layerName)) {
        return EnumerateProperties(kExtensionPropertiesCount, kExtensionProperties, pCount, pProperties);
    }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *pLayerName,
                                                                  uint32_t *pCount, VkExtensionProperties *pProperties) {
    DebugPrintf("EnumerateDeviceExtensionProperties physicalDevice %p pLayerName \"%s\"\n", physicalDevice, pLayerName);
    std::lock_guard<std::mutex> lock(global_lock);
    const auto dt = instance_dispatch_table(physicalDevice);

    if (pLayerName && !strcmp(pLayerName, kLayerProperties->layerName)) {
        return EnumerateProperties(kExtensionPropertiesCount, kExtensionProperties, pCount, pProperties);
    }
    return dt->EnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pCount, pProperties);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance instance, const char *pName) {
// Apply the DRY principle, see https://en.wikipedia.org/wiki/Don%27t_repeat_yourself
#define GET_PROC_ADDR(func) \
    if (strcmp("vk" #func, pName) == 0) return reinterpret_cast<PFN_vkVoidFunction>(func);
    GET_PROC_ADDR(GetInstanceProcAddr);
    GET_PROC_ADDR(CreateInstance);
    GET_PROC_ADDR(EnumerateInstanceLayerProperties);
    GET_PROC_ADDR(EnumerateInstanceExtensionProperties);
    GET_PROC_ADDR(EnumerateDeviceExtensionProperties);
    GET_PROC_ADDR(DestroyInstance);
    GET_PROC_ADDR(GetPhysicalDeviceProperties);
    GET_PROC_ADDR(GetPhysicalDeviceFeatures);
#undef GET_PROC_ADDR

    if (!instance) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(global_lock);
    const auto dt = instance_dispatch_table(instance);

    if (!dt->GetInstanceProcAddr) {
        return nullptr;
    }
    return dt->GetInstanceProcAddr(instance, pName);
}

}  // anonymous namespace

// Function symbols directly exported by the layer's library /////////////////////////////////////////////////////////////////////

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *pName) {
    return GetInstanceProcAddr(instance, pName);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                                                                const VkAllocationCallbacks *pAllocator, VkInstance *pInstance) {
    return CreateInstance(pCreateInfo, pAllocator, pInstance);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pCount,
                                                                                  VkLayerProperties *pProperties) {
    return EnumerateInstanceLayerProperties(pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct) {
    assert(pVersionStruct != NULL);
    assert(pVersionStruct->sType == LAYER_NEGOTIATE_INTERFACE_STRUCT);

    if (pVersionStruct->loaderLayerInterfaceVersion > CURRENT_LOADER_LAYER_INTERFACE_VERSION) {
        // Loader is requesting newer interface version; reduce to the version we support.
        pVersionStruct->loaderLayerInterfaceVersion = CURRENT_LOADER_LAYER_INTERFACE_VERSION;
    } else if (pVersionStruct->loaderLayerInterfaceVersion < CURRENT_LOADER_LAYER_INTERFACE_VERSION) {
        // Loader is requesting older interface version; record the Loader's version
        loader_layer_iface_version = pVersionStruct->loaderLayerInterfaceVersion;
    }

    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr = nullptr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    }

    return VK_SUCCESS;
}

// vim: set sw=4 ts=8 et ic ai:
