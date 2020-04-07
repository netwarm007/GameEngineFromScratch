#pragma once
#include "IRuntimeModule.hpp"
#include "ISceneParser.hpp"
#include "geommath.hpp"

namespace My {
    class SceneManager : implements IRuntimeModule
    {
    public:
        ~SceneManager() override;

        int Initialize() override;
        void Finalize() override;

        void Tick() override;

        int LoadScene(const char* scene_file_name);

        bool IsSceneChanged();
        void NotifySceneIsRenderingQueued();
        void NotifySceneIsPhysicalSimulationQueued();
        void NotifySceneIsAnimationQueued();

        const Scene& GetSceneForRendering();
        const Scene& GetSceneForPhysicalSimulation();

        void ResetScene();

        std::weak_ptr<BaseSceneNode> GetRootNode();
        std::weak_ptr<SceneGeometryNode> GetSceneGeometryNode(std::string name);
        std::weak_ptr<SceneObjectGeometry> GetSceneGeometryObject(std::string key);

    protected:
        bool LoadOgexScene(const char* ogex_scene_file_name);

    protected:
        std::shared_ptr<Scene>  m_pScene;
        bool m_bRenderingQueued = false;
        bool m_bPhysicalSimulationQueued = false;
        bool m_bAnimationQueued = false;
        bool m_bDirtyFlag = false;
    };

    extern SceneManager*    g_pSceneManager;
}

