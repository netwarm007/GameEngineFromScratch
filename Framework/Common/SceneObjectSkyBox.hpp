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
            uint32_t index = 0;

            //////////////////
            // Sky Box
            std::sprintf(filename, fmt, prefix, "posx", ext_name);
            m_Textures[index++].SetName(filename);

            std::sprintf(filename, fmt, prefix, "negx", ext_name);
            m_Textures[index++].SetName(filename);

            // we need exchange front and back when using cubemap
            // as skybox
            std::sprintf(filename, fmt, prefix, "negy", ext_name);
            m_Textures[index++].SetName(filename);

            std::sprintf(filename, fmt, prefix, "posy", ext_name);
            m_Textures[index++].SetName(filename);

            std::sprintf(filename, fmt, prefix, "posz", ext_name);
            m_Textures[index++].SetName(filename);

            std::sprintf(filename, fmt, prefix, "negz", ext_name);
            m_Textures[index++].SetName(filename);

            //////////////////
            // Irradiance Map
            std::sprintf(filename, fmt, prefix, "irradiance_posx", ext_name);
            m_Textures[index++].SetName(filename);

            std::sprintf(filename, fmt, prefix, "irradiance_negx", ext_name);
            m_Textures[index++].SetName(filename);

            // we need exchange front and back when using cubemap
            // as skybox
            std::sprintf(filename, fmt, prefix, "irradiance_negy", ext_name);
            m_Textures[index++].SetName(filename);

            std::sprintf(filename, fmt, prefix, "irradiance_posy", ext_name);
            m_Textures[index++].SetName(filename);

            std::sprintf(filename, fmt, prefix, "irradiance_posz", ext_name);
            m_Textures[index++].SetName(filename);

            std::sprintf(filename, fmt, prefix, "irradiance_negz", ext_name);
            m_Textures[index++].SetName(filename);

            //////////////////
            // Radiance Map
            const char fmt_mips[] = "%s_%s_%u_%ux%u.%s";
            uint32_t width = 512, height = 512;
            for (uint32_t mips = 0; mips < 9; mips++)
            {
                std::sprintf(filename, fmt_mips, prefix, "radiance_preview_posx", mips, width, height, ext_name);
                m_Textures[index++].SetName(filename);

                std::sprintf(filename, fmt_mips, prefix, "radiance_preview_negx", mips, width, height, ext_name);
                m_Textures[index++].SetName(filename);

                // we need exchange front and back when using cubemap
                // as skybox
                std::sprintf(filename, fmt_mips, prefix, "radiance_preview_negy", mips, width, height, ext_name);
                m_Textures[index++].SetName(filename);

                std::sprintf(filename, fmt_mips, prefix, "radiance_preview_posy", mips, width, height, ext_name);
                m_Textures[index++].SetName(filename);

                std::sprintf(filename, fmt_mips, prefix, "radiance_preview_posz", mips, width, height, ext_name);
                m_Textures[index++].SetName(filename);

                std::sprintf(filename, fmt_mips, prefix, "radiance_preview_negz", mips, width, height, ext_name);
                m_Textures[index++].SetName(filename);

                width = width >> 1;
                height = height >> 1;
            }
        }

        inline SceneObjectTexture& GetTexture(uint32_t index) 
        { 
            assert(index < 66);
            return m_Textures[index]; 
        }

        private:
            SceneObjectTexture m_Textures[66];
    };
}