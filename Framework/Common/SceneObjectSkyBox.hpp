#pragma once
#include <cstdio>

#include "SceneObjectTexture.hpp"

namespace My {
class SceneObjectSkyBox : public BaseSceneObject {
   public:
    SceneObjectSkyBox()
        : BaseSceneObject(SceneObjectType::kSceneObjectTypeSkyBox) {}
    SceneObjectSkyBox(const SceneObjectSkyBox& rhs) = delete;
    SceneObjectSkyBox(SceneObjectSkyBox&& rhs) noexcept = delete;
    SceneObjectSkyBox& operator=(const SceneObjectSkyBox& rhs) = delete;
    SceneObjectSkyBox& operator=(SceneObjectSkyBox&& rhs) = delete;

    void SetName(const char* prefix, const char* ext_name) {
        char filename[2048];
        const char fmt[] = "%s_%s.%s";
        uint32_t index = 0;

        //////////////////
        // Sky Box
        std::sprintf(filename, fmt, prefix, "posx", ext_name);
        m_Textures[index++].SetName(filename);

        std::sprintf(filename, fmt, prefix, "negx", ext_name);
        m_Textures[index++].SetName(filename);

        std::sprintf(filename, fmt, prefix, "posy", ext_name);
        m_Textures[index++].SetName(filename);

        std::sprintf(filename, fmt, prefix, "negy", ext_name);
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

        std::sprintf(filename, fmt, prefix, "irradiance_posy", ext_name);
        m_Textures[index++].SetName(filename);

        std::sprintf(filename, fmt, prefix, "irradiance_negy", ext_name);
        m_Textures[index++].SetName(filename);

        std::sprintf(filename, fmt, prefix, "irradiance_posz", ext_name);
        m_Textures[index++].SetName(filename);

        std::sprintf(filename, fmt, prefix, "irradiance_negz", ext_name);
        m_Textures[index++].SetName(filename);

        //////////////////
        // Radiance Map
        std::sprintf(filename, fmt, prefix, "radiance_posx", ext_name);
        m_Textures[index++].SetName(filename);

        std::sprintf(filename, fmt, prefix, "radiance_negx", ext_name);
        m_Textures[index++].SetName(filename);

        std::sprintf(filename, fmt, prefix, "radiance_posy", ext_name);
        m_Textures[index++].SetName(filename);

        std::sprintf(filename, fmt, prefix, "radiance_negy", ext_name);
        m_Textures[index++].SetName(filename);

        std::sprintf(filename, fmt, prefix, "radiance_posz", ext_name);
        m_Textures[index++].SetName(filename);

        std::sprintf(filename, fmt, prefix, "radiance_negz", ext_name);
        m_Textures[index++].SetName(filename);
    }

    inline SceneObjectTexture& GetTexture(uint32_t index) {
        assert(index < 18);
        return m_Textures[index];
    }

   private:
    SceneObjectTexture m_Textures[18];
};
}  // namespace My