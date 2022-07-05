#pragma once
#include "OpenGLGraphicsManagerCommonBase.hpp"

namespace My {
class OpenGLGraphicsManager : public OpenGLGraphicsManagerCommonBase {
   public:
    int Initialize() final;
    void Finalize() final;

    void CreateTextureView(Texture2D& texture_view, const TextureArrayBase& texture_array, const uint32_t layer) final;

   private:
    void getOpenGLTextureFormat(const PIXEL_FORMAT pixel_format, uint32_t& format,
                                        uint32_t& internal_format,
                                        uint32_t& type) final;

    void getOpenGLTextureFormat(const COMPRESSED_FORMAT compressed_format, uint32_t& format,
                                        uint32_t& internal_format,
                                        uint32_t& type) final;

    void BeginFrame(const Frame& frame) final;
    void EndFrame(const Frame& frame) final;
};
}  // namespace My
