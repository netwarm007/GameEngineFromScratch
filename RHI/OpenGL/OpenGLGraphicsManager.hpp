#pragma once
#include "OpenGLGraphicsManagerCommonBase.hpp"

namespace My {
    class OpenGLGraphicsManager : public OpenGLGraphicsManagerCommonBase
    {
    public:
        int Initialize() override;
    
    private:
        void getOpenGLTextureFormat(const Image& img, uint32_t& format, uint32_t& internal_format, uint32_t& type) override;
    };
}
