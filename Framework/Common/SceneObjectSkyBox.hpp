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

            std::sprintf(filename, fmt, prefix, "rt", ext_name);
            m_Textures[0].SetName(filename);

            std::sprintf(filename, fmt, prefix, "lf", ext_name);
            m_Textures[1].SetName(filename);

            // we need exchange front and back when using cubemap
            // as skybox
            std::sprintf(filename, fmt, prefix, "bk", ext_name);
            m_Textures[2].SetName(filename);

            std::sprintf(filename, fmt, prefix, "ft", ext_name);
            m_Textures[3].SetName(filename);

            std::sprintf(filename, fmt, prefix, "up", ext_name);
            m_Textures[4].SetName(filename);

            std::sprintf(filename, fmt, prefix, "dn", ext_name);
            m_Textures[5].SetName(filename);
        }

        inline SceneObjectTexture& GetRightTexture() { return m_Textures[0]; }
        inline SceneObjectTexture& GetLeftTexture()  { return m_Textures[1]; }
        inline SceneObjectTexture& GetFrontTexture() { return m_Textures[2]; }
        inline SceneObjectTexture& GetBackTexture()  { return m_Textures[3]; }
        inline SceneObjectTexture& GetTopTexture()   { return m_Textures[4]; }
        inline SceneObjectTexture& GetBottomTexture() { return m_Textures[5]; }
        inline SceneObjectTexture& GetTexture(uint32_t index) 
        { 
            assert(index < 6);
            return m_Textures[index]; 
        }

        private:
            SceneObjectTexture m_Textures[6];
    };
}