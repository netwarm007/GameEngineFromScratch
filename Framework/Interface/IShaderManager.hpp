#pragma once
#include "IRuntimeModule.hpp"
#include "portable.hpp"
#include <memory>

namespace My {
    ENUM(DefaultShaderIndex)
    {
        ShadowMap = "SHMP"_i32,
        OmniShadowMap = "OSHM"_i32,
        Basic     = "BSIC"_i32,
        Pbr       = " PBR"_i32,
        Debug     = "DEBG"_i32,
        Copy      = "COPY"_i32,
        CopyCube  = "COPC"_i32,
        CopyCube2 = "COP2"_i32,
        SkyBox    = "SKYB"_i32
    };

    Interface IShaderManager : implements IRuntimeModule
    {
    public:
        virtual ~IShaderManager() = default;

        virtual bool InitializeShaders() = 0;
        virtual void ClearShaders() = 0;

        virtual intptr_t GetDefaultShaderProgram(DefaultShaderIndex index) = 0;
    };

    extern IShaderManager* g_pShaderManager;
}