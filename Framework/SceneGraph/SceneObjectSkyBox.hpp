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

    public:
    constexpr static float skyboxVertices[]{
        1.0f,  1.0f,  1.0f,   // 0
        -1.0f, 1.0f,  1.0f,   // 1
        1.0f,  -1.0f, 1.0f,   // 2
        1.0f,  1.0f,  -1.0f,  // 3
        -1.0f, 1.0f,  -1.0f,  // 4
        1.0f,  -1.0f, -1.0f,  // 5
        -1.0f, -1.0f, 1.0f,   // 6
        -1.0f, -1.0f, -1.0f   // 7
    };

    constexpr static uint16_t skyboxIndices[]{4, 7, 5, 5, 3, 4,

                                              6, 7, 4, 4, 1, 6,

                                              5, 2, 0, 0, 3, 5,

                                              6, 1, 0, 0, 2, 6,

                                              4, 3, 0, 0, 1, 4,

                                              7, 6, 5, 5, 6, 2};

   private:
    SceneObjectTexture m_Textures[18];
};
}  // namespace My