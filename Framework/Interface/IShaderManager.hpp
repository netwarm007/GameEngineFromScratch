#pragma once
#include "IRuntimeModule.hpp"

namespace My {
    Interface IShaderManager : implements IRuntimeModule
    {
    public:
        virtual ~IShaderManager() = default;

        virtual bool InitializeShaders() = 0;
        virtual void ClearShaders() = 0;

        virtual void* GetDefaultShaderProgram() = 0;

#ifdef DEBUG
        virtual void* GetDebugShaderProgram() = 0;
#endif
    };

    extern IShaderManager* g_pShaderManager;
}