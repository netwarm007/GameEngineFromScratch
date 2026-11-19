#pragma once

#include <cstdint>

namespace My::VulkanShaderBindings {
inline constexpr uint32_t kSet = 0;
inline constexpr uint32_t kPerFrame = 10;
inline constexpr uint32_t kPerBatch = 11;
inline constexpr uint32_t kLightInfo = 12;
inline constexpr uint32_t kPassConstants = 13;
inline constexpr uint32_t kTextureBase = 32;
inline constexpr uint32_t kSamplerBase = 64;
inline constexpr uint32_t kShadowSampler = kSamplerBase + 1;
inline constexpr uint32_t kStorageImageBase = 96;
}  // namespace My::VulkanShaderBindings
