#pragma once
#include "geommath.hpp"
#include "IRuntimeModule.hpp"
#include "SceneNode.hpp"

namespace My {
    class SceneManager : implements IRuntimeModule
    {
    public:
        virtual ~SceneManager();

        virtual int Initialize();
        virtual void Finalize();

        virtual void Tick();

    protected:
        SceneEmptyNode m_RootNode;
    };
}

