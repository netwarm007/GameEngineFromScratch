#pragma once
#include "OpenGLGraphicsManagerCommonBase.hpp"

namespace My {
class OpenGLESGraphicsManager : public OpenGLGraphicsManagerCommonBase {
   public:
    int Initialize() override;

   private:
    void getOpenGLTextureFormat(const PIXEL_FORMAT pixel_format, uint32_t& format,
                                        uint32_t& internal_format,
                                        uint32_t& type) final;

    void getOpenGLTextureFormat(const COMPRESSED_FORMAT compressed_format, uint32_t& format,
                                        uint32_t& internal_format,
                                        uint32_t& type) final;
};
}  // namespace My
