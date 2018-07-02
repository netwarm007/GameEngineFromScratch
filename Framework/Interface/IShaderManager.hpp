#pragma once
#include "IRuntimeModule.hpp"
#include "portable.hpp"
#include <memory>

namespace My {
    ENUM(DefaultShaderIndex)
    {
        ShadowMap = "SHMP"_i32,
        Forward   = "FRWD"_i32,
        Differed  = "DIFR"_i32,
        Debug     = "DEBG"_i32,
        Copy      = "BTBL"_i32
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