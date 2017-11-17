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

        void LoadOgexScene(const char* scene_file_name);

    protected:
        std::unique_ptr<BaseSceneNode> m_RootNode;
    };

    extern SceneManager*    g_pSceneManager;
}

