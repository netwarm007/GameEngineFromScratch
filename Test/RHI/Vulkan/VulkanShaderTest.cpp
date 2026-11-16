#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Vulkan/VulkanShaderBindings.hpp"
#include "cbuffer.h"

using namespace My;
namespace Bindings = My::VulkanShaderBindings;

namespace {
std::atomic_uint validationErrors{0};

VKAPI_ATTR VkBool32 VKAPI_CALL DebugMessage(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        ++validationErrors;
        std::cerr << "Vulkan validation: " << data->pMessage << '\n';
    }
    return VK_FALSE;
}

void PrintAbi() {
#define FIELD(type, member) \
    std::cout << #type "_" #member "=" << offsetof(type, member) << '\n'
#define SIZE(type) std::cout << #type "_size=" << sizeof(type) << '\n'
    SIZE(PerFrameConstants);
    FIELD(PerFrameConstants, viewMatrix);
    FIELD(PerFrameConstants, projectionMatrix);
    FIELD(PerFrameConstants, camPos);
    FIELD(PerFrameConstants, numLights);
    FIELD(PerFrameConstants, clip_space_type);
    SIZE(PerBatchConstants);
    FIELD(PerBatchConstants, modelMatrix);
    SIZE(LightInfo);
    FIELD(LightInfo, lights);
    SIZE(Light);
    FIELD(Light, lightViewMatrix);
    FIELD(Light, lightProjectionMatrix);
    FIELD(Light, lightIntensity);
    FIELD(Light, lightType);
    FIELD(Light, lightCastShadow);
    FIELD(Light, lightShadowMapIndex);
    FIELD(Light, lightAngleAttenCurveType);
    FIELD(Light, lightDistAttenCurveType);
    FIELD(Light, lightSize);
    FIELD(Light, lightGuid);
    FIELD(Light, lightPosition);
    FIELD(Light, lightColor);
    FIELD(Light, lightDirection);
    FIELD(Light, lightDistAttenCurveParams);
    FIELD(Light, lightAngleAttenCurveParams);
    SIZE(DebugConstants);
    FIELD(DebugConstants, front_color);
    FIELD(DebugConstants, back_color);
    FIELD(DebugConstants, layer_index);
    FIELD(DebugConstants, mip_level);
    FIELD(DebugConstants, line_width);
    FIELD(DebugConstants, padding0);
    SIZE(ShadowMapConstants);
    FIELD(ShadowMapConstants, light_index);
    FIELD(ShadowMapConstants, shadowmap_layer_index);
    FIELD(ShadowMapConstants, near_plane);
    FIELD(ShadowMapConstants, far_plane);
    FIELD(ShadowMapConstants, shadowMatrices);
#undef FIELD
#undef SIZE
    std::cout << "binding_set=" << Bindings::kSet
              << "\nbinding_perFrame=" << Bindings::kPerFrame
              << "\nbinding_perBatch=" << Bindings::kPerBatch
              << "\nbinding_lightInfo=" << Bindings::kLightInfo
              << "\nbinding_passConstants=" << Bindings::kPassConstants
              << "\nbinding_textureBase=" << Bindings::kTextureBase
              << "\nbinding_samplerBase=" << Bindings::kSamplerBase
              << "\nbinding_storageImageBase=" << Bindings::kStorageImageBase << '\n';
}

uint32_t MemoryType(vk::PhysicalDevice physical, uint32_t bits,
                    vk::MemoryPropertyFlags flags) {
    const auto properties = physical.getMemoryProperties();
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (properties.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    throw std::runtime_error("No compatible Vulkan test memory type");
}

struct Buffer {
    vk::UniqueDeviceMemory memory;
    vk::UniqueBuffer buffer;
};

Buffer MakeBuffer(vk::Device device, vk::PhysicalDevice physical,
                  vk::DeviceSize size, vk::BufferUsageFlags usage) {
    Buffer result;
    result.buffer = device.createBufferUnique({{}, size, usage});
    const auto requirements = device.getBufferMemoryRequirements(*result.buffer);
    result.memory = device.allocateMemoryUnique({requirements.size,
        MemoryType(physical, requirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent)});
    device.bindBufferMemory(*result.buffer, *result.memory, 0);
    return result;
}

std::vector<uint32_t> ReadShader(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0 || input.tellg() % 4 != 0) {
        throw std::runtime_error("Invalid shader binary: " + path);
    }
    std::vector<uint32_t> words(static_cast<size_t>(input.tellg()) / 4);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(words.data()), words.size() * 4)) {
        throw std::runtime_error("Failed to read shader binary: " + path);
    }
    return words;
}

std::vector<float> Dispatch(vk::Device device, vk::PhysicalDevice physical,
    uint32_t family, const std::string& path, const char* entry, bool abi) {
    const uint32_t width = abi ? 2 : 512;
    const vk::DeviceSize byteSize = width * width * 4 * sizeof(float);
    auto code = ReadShader(path);
    auto shader = device.createShaderModuleUnique({{}, code.size() * 4, code.data()});
    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {Bindings::kStorageImageBase, vk::DescriptorType::eStorageImage,
         1, vk::ShaderStageFlagBits::eCompute}};
    if (abi) {
        for (auto binding : {Bindings::kPerFrame, Bindings::kPerBatch,
                             Bindings::kLightInfo, Bindings::kPassConstants}) {
            bindings.emplace_back(binding, vk::DescriptorType::eUniformBuffer,
                                  1, vk::ShaderStageFlagBits::eCompute);
        }
    }
    auto setLayout = device.createDescriptorSetLayoutUnique({{}, bindings});
    const auto rawSetLayout = *setLayout;
    auto layout = device.createPipelineLayoutUnique({{}, 1, &rawSetLayout});
    auto pipeline = device.createComputePipelineUnique({},
        {{}, {{}, vk::ShaderStageFlagBits::eCompute, *shader, entry}, *layout}).value;
    std::array<vk::DescriptorPoolSize, 2> sizes{{
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eUniformBuffer, 4}}};
    auto pool = device.createDescriptorPoolUnique({{}, 1, sizes});
    auto set = device.allocateDescriptorSets({*pool, 1, &rawSetLayout}).front();

    vk::UniqueDeviceMemory imageMemory;
    auto image = device.createImageUnique({{}, vk::ImageType::e2D,
        vk::Format::eR32G32B32A32Sfloat, {width, width, 1}, 1, 1,
        vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc});
    const auto imageRequirements = device.getImageMemoryRequirements(*image);
    imageMemory = device.allocateMemoryUnique({imageRequirements.size,
        MemoryType(physical, imageRequirements.memoryTypeBits,
                   vk::MemoryPropertyFlagBits::eDeviceLocal)});
    device.bindImageMemory(*image, *imageMemory, 0);
    auto view = device.createImageViewUnique({{}, *image, vk::ImageViewType::e2D,
        vk::Format::eR32G32B32A32Sfloat, {},
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
    vk::DescriptorImageInfo imageInfo{{}, *view, vk::ImageLayout::eGeneral};
    device.updateDescriptorSets(
        vk::WriteDescriptorSet{set, Bindings::kStorageImageBase, 0, 1,
            vk::DescriptorType::eStorageImage, &imageInfo}, {});

    PerFrameConstants frame{};
    PerBatchConstants batch{};
    LightInfo lights{};
    ShadowMapConstants shadow{};
    BuildIdentityMatrix(frame.viewMatrix);
    BuildIdentityMatrix(frame.projectionMatrix);
    BuildIdentityMatrix(batch.modelMatrix);
    batch.modelMatrix[3][0] = 4;
    frame.viewMatrix[0][0] = 2;
    frame.viewMatrix[3][1] = -3;
    frame.projectionMatrix[1][1] = 3;
    frame.camPos = {7, 8, 9, 1};
    frame.numLights = 2;
    frame.clip_space_type = 1;
    lights.lights[1].lightPosition = {-4, 5, 6, 1};
    lights.lights[1].lightIntensity = 13;
    shadow.light_index = 1;
    shadow.shadowmap_layer_index = 2;
    shadow.near_plane = 3;
    shadow.far_plane = POINT_SHADOW_FAR_PLANE;
    BuildIdentityMatrix(shadow.shadowMatrices[5]);
    shadow.shadowMatrices[5][3][0] = 0.25f;
    const std::array<const void*, 4> data{&frame, &batch, &lights, &shadow};
    const std::array<size_t, 4> lengths{
        sizeof(frame), sizeof(batch), sizeof(lights), sizeof(shadow)};
    std::vector<Buffer> uniforms;
    if (abi) {
        for (size_t i = 0; i < data.size(); ++i) {
            auto buffer = MakeBuffer(device, physical, lengths[i],
                                     vk::BufferUsageFlagBits::eUniformBuffer);
            void* mapped = device.mapMemory(*buffer.memory, 0, lengths[i]);
            std::memcpy(mapped, data[i], lengths[i]);
            device.unmapMemory(*buffer.memory);
            vk::DescriptorBufferInfo info{*buffer.buffer, 0, lengths[i]};
            device.updateDescriptorSets(vk::WriteDescriptorSet{set,
                Bindings::kPerFrame + static_cast<uint32_t>(i), 0, 1,
                vk::DescriptorType::eUniformBuffer, nullptr, &info}, {});
            uniforms.push_back(std::move(buffer));
        }
    }
    auto readback = MakeBuffer(device, physical, byteSize,
                               vk::BufferUsageFlagBits::eTransferDst);
    auto commands = device.createCommandPoolUnique({{}, family});
    auto command = device.allocateCommandBuffers(
        {*commands, vk::CommandBufferLevel::ePrimary, 1}).front();
    command.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::ImageMemoryBarrier barrier{{}, vk::AccessFlagBits::eShaderWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *image,
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, barrier);
    command.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *layout,
                               Bindings::kSet, set, {});
    command.dispatch(abi ? 1 : width, abi ? 1 : width, 1);
    barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    barrier.oldLayout = vk::ImageLayout::eGeneral;
    barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);
    command.copyImageToBuffer(*image, vk::ImageLayout::eTransferSrcOptimal,
        *readback.buffer, vk::BufferImageCopy{0, 0, 0,
            {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {width, width, 1}});
    vk::BufferMemoryBarrier hostBarrier{vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eHostRead, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, *readback.buffer, 0, VK_WHOLE_SIZE};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eHost, {}, {}, hostBarrier, {});
    command.end();
    auto fence = device.createFenceUnique({});
    device.getQueue(family, 0).submit(vk::SubmitInfo{{}, {}, command}, *fence);
    if (device.waitForFences(*fence, true, 30'000'000'000) != vk::Result::eSuccess) {
        device.waitIdle();
        throw std::runtime_error("Vulkan compute fence timed out");
    }
    std::vector<float> pixels(width * width * 4);
    const void* mapped = device.mapMemory(*readback.memory, 0, byteSize);
    std::memcpy(pixels.data(), mapped, static_cast<size_t>(byteSize));
    device.unmapMemory(*readback.memory);
    return pixels;
}

void CheckNear(float actual, float expected, float tolerance, const char* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(label) + ": expected " +
            std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

std::array<double, 2> ReferenceBrdf(uint32_t x, uint32_t y) {
    const double nv = (x + 1) / 512.0;
    const double roughness = (y + 1) / 512.0;
    const double vx = std::sqrt(1 - nv * nv);
    const double alphaSquared = std::pow(roughness, 4);
    const double k = roughness * roughness / 2;
    std::array<double, 2> integral{};
    for (uint32_t i = 0; i < 1024; ++i) {
        double radicalInverse = 0;
        double weight = 0.5;
        for (auto bits = i; bits; bits >>= 1, weight *= 0.5) {
            if (bits & 1) radicalInverse += weight;
        }
        const double phi = 2 * PI * i / 1024.0;
        const double hz = std::sqrt((1 - radicalInverse) /
            (1 + (alphaSquared - 1) * radicalInverse));
        const double hx = std::sin(phi) * std::sqrt((std::max)(0.0, 1 - hz * hz));
        const double vh = (std::max)(0.0, vx * hx + nv * hz);
        const double nl = 2 * vh * hz - nv;
        if (nl <= 0) continue;
        const double geometry = nv / (nv * (1 - k) + k) *
                                nl / (nl * (1 - k) + k);
        const double visibility = geometry * vh / (hz * nv);
        const double fresnel = std::pow(1 - vh, 5);
        integral[0] += (1 - fresnel) * visibility / 1024;
        integral[1] += fresnel * visibility / 1024;
    }
    return integral;
}

void TestCompute(const std::string& directory) {
    std::vector<const char*> layers;
    for (const auto& layer : vk::enumerateInstanceLayerProperties()) {
        if (std::string(layer.layerName.data()) == "VK_LAYER_KHRONOS_validation") {
            layers.push_back("VK_LAYER_KHRONOS_validation");
        }
    }
    if (layers.empty()) {
        throw std::runtime_error("Vulkan compute verification requires the validation layer");
    }
    std::vector<const char*> extensions{
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME};
    vk::InstanceCreateFlags instanceFlags{};
#ifdef VK_KHR_portability_enumeration
    for (const auto& extension : vk::enumerateInstanceExtensionProperties()) {
        if (std::string(extension.extensionName.data()) ==
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) {
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instanceFlags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
        }
    }
#elif defined(__APPLE__)
    throw std::runtime_error("MoltenVK verification requires headers with portability enumeration");
#endif
    VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
    debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
    debugInfo.pfnUserCallback = DebugMessage;
    VkValidationFeatureEnableEXT sync = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT validation{};
    validation.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
    validation.pNext = &debugInfo;
    validation.enabledValidationFeatureCount = 1;
    validation.pEnabledValidationFeatures = &sync;
    vk::ApplicationInfo app{"Vulkan shader regression", 1, nullptr, 0, VK_API_VERSION_1_1};
    vk::InstanceCreateInfo instanceInfo{instanceFlags, &app, layers, extensions};
    instanceInfo.pNext = &validation;
    auto instance = vk::createInstanceUnique(instanceInfo);
    auto createDebug = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        instance->getProcAddr("vkCreateDebugUtilsMessengerEXT"));
    auto destroyDebug = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        instance->getProcAddr("vkDestroyDebugUtilsMessengerEXT"));
    if (!createDebug || !destroyDebug) {
        throw std::runtime_error("Missing Vulkan debug-utils entry points");
    }
    VkDebugUtilsMessengerEXT messenger{};
    if (createDebug(*instance, &debugInfo, nullptr, &messenger) != VK_SUCCESS) {
        throw std::runtime_error("Could not create Vulkan validation messenger");
    }
    struct DebugOwner {
        VkInstance instance;
        VkDebugUtilsMessengerEXT messenger;
        PFN_vkDestroyDebugUtilsMessengerEXT destroy;
        ~DebugOwner() { destroy(instance, messenger, nullptr); }
    } debug{*instance, messenger, destroyDebug};
    for (auto physical : instance->enumeratePhysicalDevices()) {
        if (physical.getProperties().apiVersion < VK_API_VERSION_1_1 ||
            !physical.getFeatures().imageCubeArray) continue;
        if (!(physical.getFormatProperties(vk::Format::eR32G32B32A32Sfloat)
                  .optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage)) {
            continue;
        }
        auto families = physical.getQueueFamilyProperties();
        for (uint32_t family = 0; family < families.size(); ++family) {
            if (!(families[family].queueFlags & vk::QueueFlagBits::eCompute)) continue;
            const float priority = 1;
            vk::DeviceQueueCreateInfo queue{{}, family, 1, &priority};
            vk::PhysicalDeviceFeatures features{};
            features.imageCubeArray = true;
            std::vector<const char*> deviceExtensions;
            for (const auto& extension : physical.enumerateDeviceExtensionProperties()) {
                if (std::string(extension.extensionName.data()) == "VK_KHR_portability_subset") {
                    deviceExtensions.push_back("VK_KHR_portability_subset");
                }
            }
            auto device = physical.createDeviceUnique(
                {{}, queue, {}, deviceExtensions, &features});
            std::cout << "Compute device: " << physical.getProperties().deviceName
                      << "; minUniformBufferOffsetAlignment="
                      << physical.getProperties().limits.minUniformBufferOffsetAlignment << '\n';
            auto pixels = Dispatch(*device, physical, family,
                directory + "/ShaderAbi.comp.spv", "main", true);
            const std::array<float, 16> expected{
                10, -3, 3, 1, 7, 8, 9, 3, -4, 5, 6, 13,
                1 + 0.25f * POINT_SHADOW_FAR_PLANE, 2, 3, POINT_SHADOW_FAR_PLANE};
            for (size_t i = 0; i < pixels.size(); ++i) {
                CheckNear(pixels[i], expected[i], 0.0001f, "Uniform/matrix ABI");
            }
            pixels = Dispatch(*device, physical, family,
                directory + "/integrateBRDF.comp.spv", "integrateBRDF_comp_main", false);
            const std::array<std::array<uint32_t, 2>, 4> samples{{
                {63, 255}, {255, 127}, {447, 383}, {511, 511}}};
            for (const auto& sample : samples) {
                const auto i = (sample[1] * 512 + sample[0]) * 4;
                const auto reference = ReferenceBrdf(sample[0], sample[1]);
                CheckNear(pixels[i], static_cast<float>(reference[0]), 0.001f, "BRDF A");
                CheckNear(pixels[i + 1], static_cast<float>(reference[1]), 0.001f, "BRDF B");
                CheckNear(pixels[i], pixels[i + 2], 0.0001f, "BRDF RG repeat");
                CheckNear(pixels[i + 1], pixels[i + 3], 0.0001f, "BRDF BA repeat");
                std::cout << "BRDF pixel " << sample[0] << ',' << sample[1] << ": "
                          << pixels[i] << ", " << pixels[i + 1] << '\n';
            }
            device->waitIdle();
            device.reset();
            if (validationErrors != 0) {
                throw std::runtime_error("Vulkan compute produced validation errors");
            }
            return;
        }
    }
    throw std::runtime_error("No Vulkan device supports the required compute test");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Expected --abi or shader directory");
        if (std::string(argv[1]) == "--abi") {
            PrintAbi();
        } else {
            TestCompute(argv[1]);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
