#pragma once
#include "geommath.hpp"
#include "Image.hpp"
#include "IRuntimeModule.hpp"

namespace My {
    class SceneManager : implements IRuntimeModule
    {
    public:
        virtual ~SceneManager();

        virtual int Initialize();
        virtual void Finalize();

        virtual void Tick();

    protected:
        
    };
}

