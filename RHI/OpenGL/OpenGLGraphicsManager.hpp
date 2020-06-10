#pragma once
#include "OpenGLGraphicsManagerCommonBase.hpp"

namespace My {
class OpenGLGraphicsManager : public OpenGLGraphicsManagerCommonBase {
   public:
    int Initialize() final;
    void Finalize() final;

   private:
    void getOpenGLTextureFormat(const Image& img, uint32_t& format,
                                uint32_t& internal_format,
                                uint32_t& type) final;

    void BeginFrame(const Frame& frame) final;
    void EndFrame(const Frame& frame) final;
};
}  // namespace My
