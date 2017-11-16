#pragma once
#include "geommath.hpp"
#include "IRuntimeModule.hpp"
#include "SceneNode.hpp"
#include <forward_list>

namespace My {
    class SceneManager : implements IRuntimeModule
    {
    public:
        virtual ~SceneManager();

        virtual int Initialize();
        virtual void Finalize();

        virtual void Tick();

        void LoadScene(const char* scene_file_name);
        const std::shared_ptr<SceneObjectCamera> GetFirstCamera();
        const std::shared_ptr<SceneObjectCamera> GetNextCamera();
        const std::shared_ptr<SceneObjectLight> GetFirstLight();
        const std::shared_ptr<SceneObjectLight> GetNextLight();
        const std::shared_ptr<SceneObjectMaterial> GetFirstMaterial();
        const std::shared_ptr<SceneObjectMaterial> GetNextMaterial();
        const std::shared_ptr<SceneObjectGeometry> GetFirstGeometry();
        const std::shared_ptr<SceneObjectGeometry> GetNextGeometry();

    protected:
        void LoadOgexScene(const char* ogex_scene_file_name);

    protected:
        std::unique_ptr<BaseSceneNode>          m_RootNode;
        std::forward_list<std::shared_ptr<SceneObjectCamera>>      m_Cameras;
        std::forward_list<std::shared_ptr<SceneObjectLight>>       m_Lights;
        std::forward_list<std::shared_ptr<SceneObjectMaterial>>    m_Materials;
        std::forward_list<std::shared_ptr<SceneObjectGeometry>>    m_Geometries;
    };

    extern SceneManager*    g_pSceneManager;
}

