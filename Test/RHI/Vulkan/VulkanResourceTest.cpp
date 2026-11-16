#include "Vulkan/VulkanRHI.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using RHI = My::VulkanRHI;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void Bytes(const std::vector<uint8_t>& actual, const void* expected, size_t size) {
    Require(actual.size() == size && std::memcmp(actual.data(), expected, size) == 0,
            "Vulkan GPU readback differs from uploaded data");
}

struct Texture {
    RHI& rhi;
    RHI::Texture value;
    ~Texture() { rhi.DestroyTexture(value); }
};

struct Buffers {
    RHI& rhi;
    RHI::VertexBuffer vertex;
    RHI::IndexBuffer index;
    std::vector<RHI::UniformBuffer> uniform;
    ~Buffers() {
        rhi.DestroyUniformBuffers(uniform);
        const auto device = rhi.GetDevice();
        device.destroyBuffer(vertex.buffer);
        device.freeMemory(vertex.heap);
        device.destroyBuffer(index.buffer);
        device.freeMemory(index.heap);
    }
};

void CheckBuffers(RHI& rhi) {
    Buffers buffers{rhi};
    const std::array<uint32_t, 7> vertices{0x10203040, 31, 83, 257, 65537, 19, 0xabcdef01};
    const std::array<uint32_t, 3> indices{2, 0, 1};
    rhi.createVertexBuffer(vertices.data(), sizeof(vertices), buffers.vertex);
    rhi.createIndexBuffer(indices.data(), sizeof(indices), buffers.index);
    Bytes(rhi.ReadBuffer(buffers.vertex.buffer, sizeof(vertices)), vertices.data(), sizeof(vertices));
    Bytes(rhi.ReadBuffer(buffers.index.buffer, sizeof(indices)), indices.data(), sizeof(indices));

    const auto limits = rhi.GetPhysicalDevice().getProperties().limits;
    bool rejected = false;
    try {
        rhi.createUniformBuffers(buffers.uniform, 0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected && buffers.uniform.empty(), "Invalid uniform range was not rejected transactionally");
    std::array<uint32_t, 38> uniforms{};
    for (size_t i = 0; i < uniforms.size(); ++i) uniforms[i] = static_cast<uint32_t>(i * 43 + 17);
    const auto alignment = limits.minUniformBufferOffsetAlignment;
    const auto expectedStride = ((sizeof(uniforms) + alignment - 1) / alignment) * alignment;
    Require(rhi.GetUniformBufferStride(sizeof(uniforms)) == expectedStride,
            "Uniform stride does not honor the actual device alignment");
    std::vector<uint8_t> arena(static_cast<size_t>(expectedStride) * 3);
    rhi.createUniformBuffers(buffers.uniform, arena.size());
    rhi.UpdateUniformBuffer(buffers.uniform, arena.data(), arena.size());
    rhi.UpdateUniformBuffer(buffers.uniform, uniforms.data(), sizeof(uniforms));
    std::memcpy(arena.data(), uniforms.data(), sizeof(uniforms));
    for (auto& value : uniforms) value ^= 0x5a5a5a5a;
    rhi.UpdateUniformBuffer(buffers.uniform, uniforms.data(), sizeof(uniforms),
                            static_cast<size_t>(expectedStride) * 2);
    std::memcpy(arena.data() + static_cast<size_t>(expectedStride) * 2, uniforms.data(), sizeof(uniforms));
    rejected = false;
    try {
        rhi.UpdateUniformBuffer(buffers.uniform, uniforms.data(), sizeof(uniforms), arena.size());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected, "Uniform suballocation accepted a write beyond its allocation");
    Bytes(rhi.ReadBuffer(buffers.uniform[rhi.GetCurrentFrameIndex()].buffer, arena.size()),
          arena.data(), arena.size());
}

void CheckSubresources(RHI& rhi) {
    Texture array{rhi, rhi.CreateTexture(4, 4, vk::Format::eR8G8B8A8Unorm, 3, 2,
                                        vk::ImageViewType::e2DArray)};
    std::array<std::vector<uint8_t>, 2> originals;
    for (uint32_t layer = 0; layer < 2; ++layer) {
        auto& pixels = originals[layer];
        pixels.resize(64);
        for (uint32_t y = 0; y < 4; ++y) {
            for (uint32_t x = 0; x < 4; ++x) {
                const auto offset = (y * 4 + x) * 4;
                pixels[offset] = static_cast<uint8_t>(x < 2 ? 32 + layer * 16 : 192 + layer * 16);
                pixels[offset + 1] = static_cast<uint8_t>(y < 2 ? 48 - layer * 16 : 208 - layer * 16);
                pixels[offset + 2] = static_cast<uint8_t>(65 + layer * 90);
                pixels[offset + 3] = 255;
            }
        }
        rhi.UpdateTexture(array.value, pixels.data(), pixels.size(), 0, layer);
    }
    rhi.GenerateMipmaps(array.value);
    for (uint32_t layer = 0; layer < 2; ++layer) {
        const auto mip1 = rhi.ReadTexture(array.value, 1, layer);
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                Require(std::memcmp(mip1.data() + (y * 2 + x) * 4,
                                    originals[layer].data() + (y * 8 + x * 2) * 4, 4) == 0,
                        "Generated mip lost quadrant colors or array-layer independence");
            }
        }
        const auto mip2 = rhi.ReadTexture(array.value, 2, layer);
        const std::array<int, 4> expected{112 + static_cast<int>(layer) * 16,
                                         128 - static_cast<int>(layer) * 16,
                                         65 + static_cast<int>(layer) * 90, 255};
        for (size_t channel = 0; channel < 4; ++channel) {
            Require(std::abs(static_cast<int>(mip2[channel]) - expected[channel]) <= 1,
                    "Generated final mip does not match independent color averaging");
        }
        Bytes(rhi.ReadTexture(array.value, 0, layer), originals[layer].data(), 64);
    }
    const auto unchanged = rhi.ReadTexture(array.value, 1, 0);
    const std::array<uint8_t, 4> replacement{13, 57, 201, 255};
    rhi.UpdateTexture(array.value, replacement.data(), replacement.size(), 2, 1);
    Bytes(rhi.ReadTexture(array.value, 2, 1), replacement.data(), replacement.size());
    Bytes(rhi.ReadTexture(array.value, 1, 0), unchanged.data(), unchanged.size());

    Texture cubes{rhi, rhi.CreateTexture(2, 2, vk::Format::eR8G8B8A8Unorm, 1, 12,
                                        vk::ImageViewType::eCubeArray)};
    std::array<std::array<uint8_t, 16>, 12> faces{};
    for (uint32_t face = 0; face < faces.size(); ++face) {
        for (size_t i = 0; i < faces[face].size(); ++i) {
            faces[face][i] = static_cast<uint8_t>(face * 17 + i * 9);
        }
        rhi.UpdateTexture(cubes.value, faces[face].data(), faces[face].size(), 0, face);
    }
    for (uint32_t face = 0; face < faces.size(); ++face) {
        Bytes(rhi.ReadTexture(cubes.value, 0, face), faces[face].data(), faces[face].size());
    }
}

void CheckImageUploads(RHI& rhi) {
    My::Image image;
    image.Width = image.Height = 3;
    image.bitcount = 32;
    image.pixel_format = My::PIXEL_FORMAT::RGBA8;
    image.pitch = 16;
    image.data_size = 56;
    image.data = new uint8_t[image.data_size];
    std::memset(image.data, 0xcd, image.data_size);
    std::array<uint8_t, 36> expected{};
    for (size_t y = 0; y < 3; ++y) {
        for (size_t x = 0; x < 3; ++x) {
            const std::array<uint8_t, 4> color{
                static_cast<uint8_t>(10 + x * 17), static_cast<uint8_t>(20 + y * 31),
                static_cast<uint8_t>(30 + x + y * 7), 255};
            std::memcpy(image.data + y * image.pitch + x * 4, color.data(), 4);
            std::memcpy(expected.data() + (y * 3 + x) * 4, color.data(), 4);
        }
    }
    const std::array<uint8_t, 4> suppliedMip{90, 80, 70, 255};
    std::memcpy(image.data + 48, suppliedMip.data(), 4);
    image.mipmaps.emplace_back(3, 3, 16, 0, 48);
    image.mipmaps.emplace_back(1, 1, 8, 48, 8);
    Texture supplied{rhi, rhi.createTextureImage(image)};
    Bytes(rhi.ReadTexture(supplied.value), expected.data(), expected.size());
    Bytes(rhi.ReadTexture(supplied.value, 1), suppliedMip.data(), suppliedMip.size());

    image.mipmaps[1].pitch = 0;
    bool rejected = false;
    try {
        Texture invalid{rhi, rhi.createTextureImage(image)};
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected, "Invalid later-mip metadata did not fail and release its partial texture");

    My::Image rgb;
    rgb.Width = 3;
    rgb.Height = 2;
    rgb.bitcount = 24;
    rgb.pixel_format = My::PIXEL_FORMAT::RGB8;
    rgb.pitch = 12;
    rgb.data_size = 24;
    rgb.data = new uint8_t[rgb.data_size];
    std::memset(rgb.data, 0xcd, rgb.data_size);
    std::array<uint8_t, 24> rgba{};
    for (size_t y = 0; y < 2; ++y) {
        for (size_t x = 0; x < 3; ++x) {
            const std::array<uint8_t, 4> color{
                static_cast<uint8_t>(1 + x * 16), static_cast<uint8_t>(21 + y * 28),
                static_cast<uint8_t>(91 + x + y), 255};
            std::memcpy(rgb.data + y * rgb.pitch + x * 3, color.data(), 3);
            std::memcpy(rgba.data() + (y * 3 + x) * 4, color.data(), 4);
        }
    }
    Texture converted{rhi, rhi.createTextureImage(rgb)};
    Bytes(rhi.ReadTexture(converted.value), rgba.data(), rgba.size());
    rgb.is_signed = true;
    for (size_t alpha = 3; alpha < rgba.size(); alpha += 4) rgba[alpha] = 127;
    Texture signedRgb{rhi, rhi.createTextureImage(rgb)};
    Require(signedRgb.value.format == vk::Format::eR8G8B8A8Snorm,
            "Signed image metadata was lost during RGB expansion");
    Bytes(rhi.ReadTexture(signedRgb.value), rgba.data(), rgba.size());
}

void Run() {
#ifdef NDEBUG
    throw std::runtime_error("Run Vulkan resource regressions in Debug with validation enabled");
#endif
    RHI rhi;
    rhi.InitializeHeadless();
    CheckBuffers(rhi);
    CheckSubresources(rhi);
    CheckImageUploads(rhi);
    rhi.destroyAll();
    rhi.destroyAll();
    Require(rhi.GetValidationErrorCount() == 0,
            "Vulkan resource upload, mip/layer tracking, or cleanup reported validation errors");
    std::cout << "Production Vulkan buffers, alignment, array/cube mips, supplied mip chains, "
                 "RGB expansion, rollback, and GPU readback passed\n";
}
}  // namespace

int main() {
    try {
        Run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vulkan resource regression failed: " << error.what() << '\n';
        return 1;
    }
}
