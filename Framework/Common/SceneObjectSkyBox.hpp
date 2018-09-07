#pragma once
#include <cstdio>
#include "SceneObjectTexture.hpp"

namespace My {
    class SceneObjectSkyBox : public BaseSceneObject
    {
        public:
        SceneObjectSkyBox() : BaseSceneObject(SceneObjectType::kSceneObjectTypeSkyBox)
        {}

        void SetName(const char* prefix, const char* ext_name)
        {
            char filename[2048];
            const char fmt[] = "%s_%s.%s";

            std::sprintf(filename, fmt, prefix, "posx", ext_name);
            m_Textures[0].SetName(filename);

            std::sprintf(filename, fmt, prefix, "negx", ext_name);
            m_Textures[1].SetName(filename);

            // we need exchange front and back when using cubemap
            // as skybox
            std::sprintf(filename, fmt, prefix, "negy", ext_name);
            m_Textures[2].SetName(filename);

            std::sprintf(filename, fmt, prefix, "posy", ext_name);
            m_Textures[3].SetName(filename);

            std::sprintf(filename, fmt, prefix, "posz", ext_name);
            m_Textures[4].SetName(filename);

            std::sprintf(filename, fmt, prefix, "negz", ext_name);
            m_Textures[5].SetName(filename);

            std::sprintf(filename, fmt, prefix, "irradiance_posx", ext_name);
            m_Textures[6].SetName(filename);

            std::sprintf(filename, fmt, prefix, "irradiance_negx", ext_name);
            m_Textures[7].SetName(filename);

            // we need exchange front and back when using cubemap
            // as skybox
            std::sprintf(filename, fmt, prefix, "irradiance_negy", ext_name);
            m_Textures[8].SetName(filename);

            std::sprintf(filename, fmt, prefix, "irradiance_posy", ext_name);
            m_Textures[9].SetName(filename);

            std::sprintf(filename, fmt, prefix, "irradiance_posz", ext_name);
            m_Textures[10].SetName(filename);

            std::sprintf(filename, fmt, prefix, "irradiance_negz", ext_name);
            m_Textures[11].SetName(filename);
        }

        inline SceneObjectTexture& GetRightTexture() { return m_Textures[0]; }
        inline SceneObjectTexture& GetLeftTexture()  { return m_Textures[1]; }
        inline SceneObjectTexture& GetFrontTexture() { return m_Textures[2]; }
        inline SceneObjectTexture& GetBackTexture()  { return m_Textures[3]; }
        inline SceneObjectTexture& GetTopTexture()   { return m_Textures[4]; }
        inline SceneObjectTexture& GetBottomTexture() { return m_Textures[5]; }

        inline SceneObjectTexture& GetRightIrradianceTexture() { return m_Textures[6]; }
        inline SceneObjectTexture& GetLeftIrradianceTexture()  { return m_Textures[7]; }
        inline SceneObjectTexture& GetFrontIrradianceTexture() { return m_Textures[8]; }
        inline SceneObjectTexture& GetBackIrradianceTexture()  { return m_Textures[9]; }
        inline SceneObjectTexture& GetTopIrradianceTexture()   { return m_Textures[10]; }
        inline SceneObjectTexture& GetBottomIrradianceTexture() { return m_Textures[11]; }

        inline SceneObjectTexture& GetTexture(uint32_t index) 
        { 
            assert(index < 12);
            return m_Textures[index]; 
        }

        private:
            SceneObjectTexture m_Textures[12];
    };
}