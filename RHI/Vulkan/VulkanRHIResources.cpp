#include "VulkanRHI.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace My;

namespace {
struct Block {
    uint32_t width;
    uint32_t height;
    uint32_t bytes;
};

Block FormatBlock(vk::Format format) {
    switch (format) {
        case vk::Format::eR8Unorm:
        case vk::Format::eR8Snorm:
        case vk::Format::eR8Uint:
        case vk::Format::eR8Sint: return {1, 1, 1};
        case vk::Format::eR8G8Unorm:
        case vk::Format::eR8G8Snorm:
        case vk::Format::eR16Unorm:
        case vk::Format::eR16Snorm:
        case vk::Format::eR16Sfloat:
        case vk::Format::eD16Unorm:
        case vk::Format::eR5G6B5UnormPack16: return {1, 1, 2};
        case vk::Format::eR8G8B8Unorm: return {1, 1, 3};
        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eR8G8B8A8Srgb:
        case vk::Format::eR8G8B8A8Snorm:
        case vk::Format::eB8G8R8A8Unorm:
        case vk::Format::eB8G8R8A8Srgb:
        case vk::Format::eR16G16Unorm:
        case vk::Format::eR16G16Snorm:
        case vk::Format::eR16G16Sfloat:
        case vk::Format::eR32Sfloat:
        case vk::Format::eR32Uint:
        case vk::Format::eR32Sint:
        case vk::Format::eD32Sfloat:
        case vk::Format::eA2B10G10R10UnormPack32: return {1, 1, 4};
        case vk::Format::eR16G16B16Unorm:
        case vk::Format::eR16G16B16Snorm:
        case vk::Format::eR16G16B16Sfloat: return {1, 1, 6};
        case vk::Format::eR16G16B16A16Unorm:
        case vk::Format::eR16G16B16A16Snorm:
        case vk::Format::eR16G16B16A16Sfloat:
        case vk::Format::eR32G32Sfloat:
        case vk::Format::eR32G32Uint:
        case vk::Format::eR32G32Sint: return {1, 1, 8};
        case vk::Format::eR32G32B32Sfloat:
        case vk::Format::eR32G32B32Uint:
        case vk::Format::eR32G32B32Sint: return {1, 1, 12};
        case vk::Format::eR32G32B32A32Sfloat:
        case vk::Format::eR32G32B32A32Uint:
        case vk::Format::eR32G32B32A32Sint: return {1, 1, 16};
        case vk::Format::eBc1RgbUnormBlock:
        case vk::Format::eBc1RgbSrgbBlock:
        case vk::Format::eBc1RgbaUnormBlock:
        case vk::Format::eBc1RgbaSrgbBlock:
        case vk::Format::eBc4UnormBlock:
        case vk::Format::eBc4SnormBlock: return {4, 4, 8};
        case vk::Format::eBc2UnormBlock:
        case vk::Format::eBc2SrgbBlock:
        case vk::Format::eBc3UnormBlock:
        case vk::Format::eBc3SrgbBlock:
        case vk::Format::eBc5UnormBlock:
        case vk::Format::eBc5SnormBlock:
        case vk::Format::eBc6HUfloatBlock:
        case vk::Format::eBc6HSfloatBlock:
        case vk::Format::eBc7UnormBlock:
        case vk::Format::eBc7SrgbBlock: return {4, 4, 16};
        case vk::Format::eAstc4x4SrgbBlock: return {4, 4, 16};
        case vk::Format::eAstc5x4SrgbBlock: return {5, 4, 16};
        case vk::Format::eAstc5x5SrgbBlock: return {5, 5, 16};
        case vk::Format::eAstc6x5SrgbBlock: return {6, 5, 16};
        case vk::Format::eAstc6x6SrgbBlock: return {6, 6, 16};
        case vk::Format::eAstc8x5SrgbBlock: return {8, 5, 16};
        case vk::Format::eAstc8x6SrgbBlock: return {8, 6, 16};
        case vk::Format::eAstc8x8SrgbBlock: return {8, 8, 16};
        case vk::Format::eAstc10x5SrgbBlock: return {10, 5, 16};
        case vk::Format::eAstc10x6SrgbBlock: return {10, 6, 16};
        case vk::Format::eAstc10x8SrgbBlock: return {10, 8, 16};
        case vk::Format::eAstc10x10SrgbBlock: return {10, 10, 16};
        case vk::Format::eAstc12x10SrgbBlock: return {12, 10, 16};
        case vk::Format::eAstc12x12SrgbBlock: return {12, 12, 16};
        default:
            throw std::invalid_argument("Vulkan texture upload/readback format unsupported: " +
                                        vk::to_string(format));
    }
}

vk::Extent3D MipExtent(const VulkanRHI::Texture& texture, uint32_t mip) {
    return {(std::max)(1u, texture.width >> mip),
            (std::max)(1u, texture.height >> mip), 1};
}

size_t SubresourceBytes(const VulkanRHI::Texture& texture, uint32_t mip) {
    const auto block = FormatBlock(texture.format);
    const auto extent = MipExtent(texture, mip);
    const uint64_t row = ((uint64_t{extent.width} + block.width - 1) / block.width) * block.bytes;
    const uint64_t rows = (uint64_t{extent.height} + block.height - 1) / block.height;
    if (row > (std::numeric_limits<size_t>::max)() / rows) {
        throw std::overflow_error("Vulkan texture byte count overflow");
    }
    return static_cast<size_t>(row * rows);
}

void CheckSubresource(const VulkanRHI::Texture& texture, uint32_t mip, uint32_t layer) {
    if (!texture.image || mip >= texture.mipLevels || layer >= texture.layers ||
        texture.layouts.size() != size_t{texture.mipLevels} * texture.layers) {
        throw std::invalid_argument("Invalid Vulkan texture subresource");
    }
}

vk::ImageSubresourceRange Subresource(const VulkanRHI::Texture& texture,
                                     uint32_t mip, uint32_t layer) {
    return {texture.aspect, mip, 1, layer, 1};
}

vk::ImageLayout ReadyLayout(const VulkanRHI::Texture& texture) {
    return (texture.usage & vk::ImageUsageFlagBits::eSampled) &&
                   !(texture.usage & vk::ImageUsageFlagBits::eStorage)
               ? vk::ImageLayout::eShaderReadOnlyOptimal : vk::ImageLayout::eGeneral;
}

std::pair<vk::PipelineStageFlags, vk::AccessFlags> LayoutScope(vk::ImageLayout layout) {
    using Stage = vk::PipelineStageFlagBits;
    using Access = vk::AccessFlagBits;
    switch (layout) {
        case vk::ImageLayout::eUndefined: return {Stage::eTopOfPipe, {}};
        case vk::ImageLayout::eTransferDstOptimal: return {Stage::eTransfer, Access::eTransferWrite};
        case vk::ImageLayout::eTransferSrcOptimal: return {Stage::eTransfer, Access::eTransferRead};
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return {Stage::eVertexShader | Stage::eFragmentShader | Stage::eComputeShader,
                    Access::eShaderRead | Access::eInputAttachmentRead};
        case vk::ImageLayout::eGeneral:
            return {Stage::eAllCommands, Access::eMemoryRead | Access::eMemoryWrite};
        case vk::ImageLayout::eColorAttachmentOptimal:
            return {Stage::eColorAttachmentOutput, Access::eColorAttachmentRead | Access::eColorAttachmentWrite};
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            return {Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
                    Access::eDepthStencilAttachmentRead | Access::eDepthStencilAttachmentWrite};
        case vk::ImageLayout::eDepthStencilReadOnlyOptimal:
            return {Stage::eEarlyFragmentTests | Stage::eLateFragmentTests |
                        Stage::eVertexShader | Stage::eFragmentShader | Stage::eComputeShader,
                    Access::eDepthStencilAttachmentRead | Access::eShaderRead | Access::eInputAttachmentRead};
        default: throw std::invalid_argument("Unsupported Vulkan resource layout transition");
    }
}

struct TextureOwner {
    vk::Device device;
    VulkanRHI::Texture& texture;
    bool active = true;
    ~TextureOwner() {
        if (active) {
            device.destroyImageView(texture.descriptor);
            device.destroyImage(texture.image);
            device.freeMemory(texture.heap);
        }
    }
};
}  // namespace

vk::DeviceSize VulkanRHI::GetUniformBufferStride(vk::DeviceSize size) const {
    if (!m_vkPhysicalDevice) throw std::logic_error("Uniform alignment requires a Vulkan device");
    const auto alignment = m_vkPhysicalDevice.getProperties().limits.minUniformBufferOffsetAlignment;
    if (size == 0 || size > (std::numeric_limits<vk::DeviceSize>::max)() - (alignment - 1)) {
        throw std::invalid_argument("Invalid Vulkan uniform allocation size");
    }
    return ((size + alignment - 1) / alignment) * alignment;
}

std::vector<uint8_t> VulkanRHI::ReadBuffer(vk::Buffer buffer, vk::DeviceSize size) {
    if (!m_vkDevice || !buffer || size == 0 || size > (std::numeric_limits<size_t>::max)()) {
        throw std::invalid_argument("Invalid Vulkan buffer readback");
    }
    std::vector<uint8_t> result(static_cast<size_t>(size));
    BufferAllocation staging(m_vkDevice);
    createBuffer(size, vk::BufferUsageFlagBits::eTransferDst,
                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                  staging.buffer, staging.memory);
    const auto command = beginSingleTimeCommands();
    const vk::MemoryBarrier readable(vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eTransferRead);
    command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                             vk::PipelineStageFlagBits::eTransfer, {}, readable, {}, {});
    command.copyBuffer(buffer, staging.buffer, vk::BufferCopy(0, 0, size));
    const vk::MemoryBarrier host(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead);
    command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                             vk::PipelineStageFlagBits::eHost, {}, host, {}, {});
    auto submitted = command;
    endSingleTimeCommands(submitted);
    const auto data = m_vkDevice.mapMemory(staging.memory, 0, size);
    std::memcpy(result.data(), data, result.size());
    m_vkDevice.unmapMemory(staging.memory);
    return result;
}

VulkanRHI::Texture VulkanRHI::CreateTexture(
    uint32_t width, uint32_t height, vk::Format format, uint32_t mipLevels,
    uint32_t layers, vk::ImageViewType viewType, vk::ImageUsageFlags usage,
    vk::SampleCountFlagBits samples) {
    if (!m_vkDevice) throw std::logic_error("Texture creation requires a Vulkan device");
    uint32_t maxMips = 1;
    for (auto dimension = (std::max)(width, height); dimension > 1; dimension >>= 1) ++maxMips;
    const bool cube = viewType == vk::ImageViewType::eCube || viewType == vk::ImageViewType::eCubeArray;
    if (width == 0 || height == 0 || layers == 0 || mipLevels == 0 || mipLevels > maxMips ||
        format == vk::Format::eUndefined ||
        (viewType != vk::ImageViewType::e2D && viewType != vk::ImageViewType::e2DArray && !cube) ||
        (viewType == vk::ImageViewType::e2D && layers != 1) ||
        (viewType == vk::ImageViewType::eCube && layers != 6) ||
        (cube && (width != height || layers % 6 != 0)) ||
        (samples != vk::SampleCountFlagBits::e1 && (mipLevels != 1 || cube)) ||
        !(static_cast<uint32_t>(samples)) ||
        (static_cast<uint32_t>(samples) & (static_cast<uint32_t>(samples) - 1))) {
        throw std::invalid_argument("Invalid Vulkan texture dimensions, format, or view type");
    }
    const auto features = m_vkPhysicalDevice.getFeatures();
    if ((format >= vk::Format::eBc1RgbUnormBlock && format <= vk::Format::eBc7SrgbBlock &&
         !features.textureCompressionBC) ||
        (format >= vk::Format::eEtc2R8G8B8UnormBlock && format <= vk::Format::eEacR11G11SnormBlock &&
         !features.textureCompressionETC2) ||
        (format >= vk::Format::eAstc4x4UnormBlock && format <= vk::Format::eAstc12x12SrgbBlock &&
         !features.textureCompressionASTC_LDR)) {
        throw std::runtime_error("Vulkan texture compression feature unavailable for " + vk::to_string(format));
    }
    Texture texture;
    texture.width = width;
    texture.height = height;
    texture.mipLevels = mipLevels;
    texture.layers = layers;
    texture.format = format;
    texture.samples = samples;
    texture.aspect = vk::ImageAspectFlagBits::eColor;
    if (format == vk::Format::eD16Unorm || format == vk::Format::eD32Sfloat ||
        format == vk::Format::eX8D24UnormPack32) {
        texture.aspect = vk::ImageAspectFlagBits::eDepth;
    } else if (format == vk::Format::eD24UnormS8Uint || format == vk::Format::eD32SfloatS8Uint ||
               format == vk::Format::eD16UnormS8Uint) {
        texture.aspect = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    } else if (format == vk::Format::eS8Uint) {
        texture.aspect = vk::ImageAspectFlagBits::eStencil;
    }
    texture.usage = usage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
    const vk::ImageCreateFlags flags = cube ? vk::ImageCreateFlagBits::eCubeCompatible : vk::ImageCreateFlags{};
    VkImageFormatProperties support{};
    const auto supported = vkGetPhysicalDeviceImageFormatProperties(
        m_vkPhysicalDevice, static_cast<VkFormat>(format), VK_IMAGE_TYPE_2D,
        VK_IMAGE_TILING_OPTIMAL, static_cast<VkImageUsageFlags>(texture.usage),
        static_cast<VkImageCreateFlags>(flags), &support);
    if (supported != VK_SUCCESS) {
        throw std::runtime_error("Vulkan texture format/usage query failed for " +
                                  vk::to_string(format) + ": " +
                                  vk::to_string(static_cast<vk::Result>(supported)));
    }
    if (width > support.maxExtent.width || height > support.maxExtent.height ||
        layers > support.maxArrayLayers || mipLevels > support.maxMipLevels ||
        !(support.sampleCounts & static_cast<VkSampleCountFlags>(samples))) {
        throw std::invalid_argument("Vulkan texture exceeds device image limits");
    }
    texture.layouts.assign(size_t{mipLevels} * layers, vk::ImageLayout::eUndefined);
    TextureOwner owner{m_vkDevice, texture};
    const vk::ImageCreateInfo image(flags, vk::ImageType::e2D, format,
        vk::Extent3D(width, height, 1), mipLevels, layers, samples,
        vk::ImageTiling::eOptimal, texture.usage);
    texture.image = m_vkDevice.createImage(image);
    const auto memory = m_vkDevice.getImageMemoryRequirements(texture.image);
    texture.heap = m_vkDevice.allocateMemory({memory.size,
        findMemoryType(memory.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)});
    m_vkDevice.bindImageMemory(texture.image, texture.heap, 0);
    const vk::ImageViewCreateInfo view({}, texture.image, viewType, format, {},
        vk::ImageSubresourceRange(texture.aspect, 0, mipLevels, 0, layers));
    texture.descriptor = m_vkDevice.createImageView(view);
    owner.active = false;
    return texture;
}

void VulkanRHI::TransitionTexture(vk::CommandBuffer command, Texture& texture,
                                  const vk::ImageSubresourceRange& range,
                                  vk::ImageLayout layout) {
    if (!command || layout == vk::ImageLayout::eUndefined ||
        range.aspectMask != texture.aspect || range.levelCount == 0 || range.layerCount == 0 ||
        range.baseMipLevel >= texture.mipLevels || range.levelCount > texture.mipLevels - range.baseMipLevel ||
        range.baseArrayLayer >= texture.layers || range.layerCount > texture.layers - range.baseArrayLayer) {
        throw std::invalid_argument("Invalid Vulkan texture transition range");
    }
    CheckSubresource(texture, range.baseMipLevel, range.baseArrayLayer);
    CheckTextureAvailable(texture, command);
    const auto destination = LayoutScope(layout);
    using Layout = vk::ImageLayout;
    using Usage = vk::ImageUsageFlagBits;
    vk::ImageUsageFlags requiredUsage;
    if (layout == Layout::eShaderReadOnlyOptimal) requiredUsage = Usage::eSampled | Usage::eInputAttachment;
    if (layout == Layout::eColorAttachmentOptimal) requiredUsage = Usage::eColorAttachment;
    if (layout == Layout::eDepthStencilAttachmentOptimal) requiredUsage = Usage::eDepthStencilAttachment;
    if (layout == Layout::eDepthStencilReadOnlyOptimal) {
        requiredUsage = Usage::eDepthStencilAttachment | Usage::eSampled | Usage::eInputAttachment;
    }
    if (layout == Layout::eTransferSrcOptimal) requiredUsage = Usage::eTransferSrc;
    if (layout == Layout::eTransferDstOptimal) requiredUsage = Usage::eTransferDst;
    if ((requiredUsage && !(texture.usage & requiredUsage)) ||
        (layout == Layout::eColorAttachmentOptimal && texture.aspect != vk::ImageAspectFlagBits::eColor) ||
        ((layout == Layout::eDepthStencilAttachmentOptimal || layout == Layout::eDepthStencilReadOnlyOptimal) &&
         texture.aspect == vk::ImageAspectFlagBits::eColor)) {
        throw std::invalid_argument("Vulkan texture layout is incompatible with its usage or aspect");
    }
    if (m_commands && m_commands->Get() == command) {
        m_commands->m_layouts.try_emplace(&texture, texture.layouts);
    } else if (m_frameActive && GetFrameCommandBuffer() == command) {
        m_frameLayouts.try_emplace(&texture, texture.layouts);
    }
    for (uint32_t layer = range.baseArrayLayer; layer < range.baseArrayLayer + range.layerCount; ++layer) {
        for (uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.levelCount; ++mip) {
            auto& previous = texture.layouts[size_t{layer} * texture.mipLevels + mip];
            if (previous == layout) continue;
            const auto source = LayoutScope(previous);
            const vk::ImageMemoryBarrier barrier(source.second, destination.second, previous, layout,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, texture.image,
                vk::ImageSubresourceRange(texture.aspect, mip, 1, layer, 1));
            command.pipelineBarrier(source.first, destination.first, {}, {}, {}, barrier);
            previous = layout;
        }
    }
}

void VulkanRHI::UpdateTexture(Texture& texture, const void* data, size_t size,
                              uint32_t mip, uint32_t layer) {
    CheckSubresource(texture, mip, layer);
    if (!data || size != SubresourceBytes(texture, mip) ||
        texture.samples != vk::SampleCountFlagBits::e1 ||
        (texture.aspect & vk::ImageAspectFlagBits::eStencil)) {
        throw std::invalid_argument("Vulkan texture upload requires one tightly packed mip/layer");
    }
    BufferAllocation staging(m_vkDevice);
    createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                  staging.buffer, staging.memory);
    auto mapped = m_vkDevice.mapMemory(staging.memory, 0, size);
    std::memcpy(mapped, data, size);
    m_vkDevice.unmapMemory(staging.memory);
    ExecuteTextureCommands(texture, [&](vk::CommandBuffer command) {
        const auto range = Subresource(texture, mip, layer);
        TransitionTexture(command, texture, range, vk::ImageLayout::eTransferDstOptimal);
        const vk::BufferImageCopy copy(0, 0, 0,
            vk::ImageSubresourceLayers(texture.aspect, mip, layer, 1), {0, 0, 0}, MipExtent(texture, mip));
        command.copyBufferToImage(staging.buffer, texture.image, vk::ImageLayout::eTransferDstOptimal, copy);
        TransitionTexture(command, texture, range, ReadyLayout(texture));
    });
}

std::vector<uint8_t> VulkanRHI::ReadTexture(Texture& texture, uint32_t mip, uint32_t layer) {
    CheckSubresource(texture, mip, layer);
    const auto previous = texture.layouts[size_t{layer} * texture.mipLevels + mip];
    if (previous == vk::ImageLayout::eUndefined || texture.samples != vk::SampleCountFlagBits::e1 ||
        (texture.aspect & vk::ImageAspectFlagBits::eStencil)) {
        throw std::invalid_argument("Cannot read undefined or packed depth/stencil texture contents");
    }
    std::vector<uint8_t> result(SubresourceBytes(texture, mip));
    BufferAllocation staging(m_vkDevice);
    createBuffer(result.size(), vk::BufferUsageFlagBits::eTransferDst,
                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                  staging.buffer, staging.memory);
    ExecuteTextureCommands(texture, [&](vk::CommandBuffer command) {
        const auto range = Subresource(texture, mip, layer);
        TransitionTexture(command, texture, range, vk::ImageLayout::eTransferSrcOptimal);
        const vk::BufferImageCopy copy(0, 0, 0,
            vk::ImageSubresourceLayers(texture.aspect, mip, layer, 1), {0, 0, 0}, MipExtent(texture, mip));
        command.copyImageToBuffer(texture.image, vk::ImageLayout::eTransferSrcOptimal, staging.buffer, copy);
        const vk::MemoryBarrier host(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead);
        command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                 vk::PipelineStageFlagBits::eHost, {}, host, {}, {});
        TransitionTexture(command, texture, range, previous);
    });
    auto mapped = m_vkDevice.mapMemory(staging.memory, 0, result.size());
    std::memcpy(result.data(), mapped, result.size());
    m_vkDevice.unmapMemory(staging.memory);
    return result;
}

void VulkanRHI::GenerateMipmaps(Texture& texture) {
    CheckSubresource(texture, 0, 0);
    if (texture.samples != vk::SampleCountFlagBits::e1) {
        throw std::invalid_argument("Vulkan multisample images cannot have mipmaps");
    }
    if (texture.mipLevels == 1) return;
    const auto required = vk::FormatFeatureFlagBits::eBlitSrc |
                          vk::FormatFeatureFlagBits::eBlitDst |
                          vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
    const auto features = m_vkPhysicalDevice.getFormatProperties(texture.format).optimalTilingFeatures;
    if ((features & required) != required || texture.aspect != vk::ImageAspectFlagBits::eColor) {
        throw std::runtime_error("Linear Vulkan mip generation unsupported for " + vk::to_string(texture.format));
    }
    for (uint32_t layer = 0; layer < texture.layers; ++layer) {
        if (texture.layouts[size_t{layer} * texture.mipLevels] == vk::ImageLayout::eUndefined) {
            throw std::logic_error("Vulkan mip generation requires initialized base levels");
        }
    }
    ExecuteTextureCommands(texture, [&](vk::CommandBuffer command) {
    for (uint32_t layer = 0; layer < texture.layers; ++layer) {
        for (uint32_t mip = 1; mip < texture.mipLevels; ++mip) {
            TransitionTexture(command, texture, Subresource(texture, mip - 1, layer), vk::ImageLayout::eTransferSrcOptimal);
            TransitionTexture(command, texture, Subresource(texture, mip, layer), vk::ImageLayout::eTransferDstOptimal);
            const auto from = MipExtent(texture, mip - 1);
            const auto to = MipExtent(texture, mip);
            vk::ImageBlit blit;
            blit.srcSubresource = vk::ImageSubresourceLayers(texture.aspect, mip - 1, layer, 1);
            blit.srcOffsets[1] = vk::Offset3D(from.width, from.height, 1);
            blit.dstSubresource = vk::ImageSubresourceLayers(texture.aspect, mip, layer, 1);
            blit.dstOffsets[1] = vk::Offset3D(to.width, to.height, 1);
            command.blitImage(texture.image, vk::ImageLayout::eTransferSrcOptimal,
                              texture.image, vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);
            TransitionTexture(command, texture, Subresource(texture, mip - 1, layer), ReadyLayout(texture));
        }
        TransitionTexture(command, texture, Subresource(texture, texture.mipLevels - 1, layer), ReadyLayout(texture));
    }
    });
}

void VulkanRHI::DestroyTexture(Texture& texture) {
    CheckTextureAvailable(texture);
    if (!m_vkDevice) {
        if (texture.image || texture.descriptor || texture.heap) {
            throw std::logic_error("Vulkan texture outlived its device");
        }

    } else {
        m_vkDevice.destroyImageView(texture.descriptor);
        m_vkDevice.destroyImage(texture.image);
        m_vkDevice.freeMemory(texture.heap);
    }
    texture = {};
}

void VulkanRHI::ResolveTexture(vk::CommandBuffer command, Texture& source, Texture& target) {
    if (source.samples == vk::SampleCountFlagBits::e1 ||
        target.samples != vk::SampleCountFlagBits::e1 ||
        source.aspect != vk::ImageAspectFlagBits::eColor || source.aspect != target.aspect ||
        source.format != target.format || source.width != target.width || source.height != target.height ||
        source.layers != 1 || target.layers != 1 ||
        source.layouts.empty() || source.layouts[0] == vk::ImageLayout::eUndefined) {
        throw std::invalid_argument("Vulkan resolve requires matching initialized multisample/color targets");
    }
    const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    TransitionTexture(command, source, range, vk::ImageLayout::eTransferSrcOptimal);
    TransitionTexture(command, target, range, vk::ImageLayout::eTransferDstOptimal);
    const vk::ImageSubresourceLayers layer(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
    command.resolveImage(source.image, vk::ImageLayout::eTransferSrcOptimal, target.image,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageResolve(layer, {0, 0, 0}, layer, {0, 0, 0}, {source.width, source.height, 1}));
    TransitionTexture(command, source, range, vk::ImageLayout::eColorAttachmentOptimal);
    TransitionTexture(command, target, range, vk::ImageLayout::eShaderReadOnlyOptimal);
}

VulkanRHI::Texture VulkanRHI::createTextureImage(Image& image) {
    if (!image.data || image.Width == 0 || image.Height == 0 || image.mipmaps.size() > 32) {
        throw std::invalid_argument("Invalid Vulkan image upload");
    }
    vk::Format format;
    getTextureFormat(image, format);
    const bool expandRgb = !image.compressed && image.bitcount == 24;
    const auto block = expandRgb ? Block{1, 1, 3} : FormatBlock(format);
    const auto levels = image.mipmaps.empty()
        ? std::vector<Image::Mipmap>{Image::Mipmap(image.Width, image.Height, image.pitch, 0, image.data_size)}
        : image.mipmaps;
    auto texture = CreateTexture(image.Width, image.Height, format, static_cast<uint32_t>(levels.size()));
    TextureOwner owner{m_vkDevice, texture};
    for (uint32_t mip = 0; mip < levels.size(); ++mip) {
        const auto& source = levels[mip];
        const auto extent = MipExtent(texture, mip);
        const uint64_t rowBytes = ((uint64_t{source.Width} + block.width - 1) / block.width) * block.bytes;
        const uint64_t rows = (uint64_t{source.Height} + block.height - 1) / block.height;
        if (source.Width != extent.width || source.Height != extent.height ||
            source.pitch < rowBytes || source.offset > image.data_size ||
            source.data_size > image.data_size - source.offset ||
            rows == 0 || rowBytes > source.data_size ||
            rows - 1 > (source.data_size - rowBytes) / source.pitch) {
            throw std::invalid_argument("Invalid Vulkan image mip dimensions, pitch, or byte range");
        }
        std::vector<uint8_t> packed(SubresourceBytes(texture, mip));
        for (size_t row = 0; row < rows; ++row) {
            const auto* input = image.data + source.offset + row * source.pitch;
            if (expandRgb) {
                for (uint32_t x = 0; x < source.Width; ++x) {
                    const auto output = (row * source.Width + x) * 4;
                    std::memcpy(packed.data() + output, input + size_t{x} * 3, 3);
                    packed[output + 3] = image.is_signed ? 127 : 255;
                }
            } else {
                std::memcpy(packed.data() + row * rowBytes, input, static_cast<size_t>(rowBytes));
            }
        }
        UpdateTexture(texture, packed.data(), packed.size(), mip);
    }
    owner.active = false;
    return texture;
}

void VulkanRHI::DestroyUniformBuffers(std::vector<UniformBuffer>& buffers) {
    if (!m_vkDevice && !buffers.empty()) throw std::logic_error("Vulkan uniforms outlived their device");
    for (auto& buffer : buffers) {
        m_vkDevice.destroyBuffer(buffer.buffer);
        m_vkDevice.freeMemory(buffer.heap);
    }
    buffers.clear();
}
