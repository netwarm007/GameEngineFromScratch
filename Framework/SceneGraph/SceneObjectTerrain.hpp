#pragma once
#include <cstdio>

#include "SceneObjectTexture.hpp"

namespace My {
class SceneObjectTerrain : public BaseSceneObject {
   public:
    SceneObjectTerrain()
        : BaseSceneObject(SceneObjectType::kSceneObjectTypeTerrain) {}

    void SetName(const char* prefix, const char* ext_name) {
        const size_t FILENAME_SIZE = 2048;
        char filename[FILENAME_SIZE];
        const char fmt[] = "%s_%d_%d.%s";
        uint32_t index = 0;

        for (int i = 0; i < nMaxTerrainGridWidth; i++) {
            for (int j = 0; j < nMaxTerrainGridHeight; j++) {
                std::snprintf(filename, FILENAME_SIZE, fmt, prefix, i, j, ext_name);
                m_Textures[index++].SetName(filename);
            }
        }

        assert(index == nMaxTerrainHeightMapCount);
    }

    inline SceneObjectTexture& GetTexture(uint32_t index) {
        assert(index < nMaxTerrainHeightMapCount);
        return m_Textures[index];
    }

   private:
    static const int32_t nMaxTerrainGridWidth = 16;
    static const int32_t nMaxTerrainGridHeight = 16;
    static const int32_t nMaxTerrainHeightMapCount =
        nMaxTerrainGridWidth * nMaxTerrainGridHeight;

    SceneObjectTexture m_Textures[nMaxTerrainHeightMapCount];
};
}  // namespace My