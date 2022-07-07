#pragma once
#include "OpenGLGraphicsManagerCommonBase.hpp"

namespace My {
class OpenGLGraphicsManager : public OpenGLGraphicsManagerCommonBase {
   public:
    int Initialize() final;
    void Finalize() final;

    void CreateTextureView(Texture2D& texture_view, const TextureArrayBase& texture_array, const uint32_t slice, const uint32_t mip) final;

    void BeginPass(Frame& frame) final;
    void EndPass(Frame& frame) final;

   private:
    void getOpenGLTextureFormat(const PIXEL_FORMAT pixel_format, uint32_t& format,
                                        uint32_t& internal_format,
                                        uint32_t& type) final;

    void getOpenGLTextureFormat(const COMPRESSED_FORMAT compressed_format, uint32_t& format,
                                        uint32_t& internal_format,
                                        uint32_t& type) final;

    void BeginFrame(Frame& frame) final;
    void EndFrame(Frame& frame) final;
};
}  // namespace My
