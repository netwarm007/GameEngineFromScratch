#pragma once
#include <future>
#include <utility>

#include "AssetLoader.hpp"
#include "BMP.hpp"
#include "BaseSceneObject.hpp"
#include "DDS.hpp"
#include "HDR.hpp"
#include "JPEG.hpp"
#include "PNG.hpp"
#include "TGA.hpp"
#include "geommath.hpp"

namespace My {
class SceneObjectTexture : public BaseSceneObject {
   private:
    std::string m_Name;
    uint32_t m_nTexCoordIndex{0};
    std::vector<Matrix4X4f> m_Transforms;
    std::shared_ptr<Image> m_pImage;
    std::future<bool> m_asyncLoadFuture;

   public:
    SceneObjectTexture()
        : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture) {}
    explicit SceneObjectTexture(const std::string& name)
        : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture),
          m_Name(name) {
        LoadTextureAsync();
    }

    void AddTransform(Matrix4X4f& matrix) { m_Transforms.push_back(matrix); }
    void SetName(const std::string& name) {
        m_Name = name;
        LoadTextureAsync();
    }
    void SetName(std::string&& name) {
        m_Name = std::forward<std::string>(name);
        LoadTextureAsync();
    }
    [[nodiscard]] const std::string& GetName() const { return m_Name; }

    std::shared_ptr<Image> GetTextureImage();

   private:
    bool LoadTexture();
    void LoadTextureAsync();

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectTexture& obj);
};
}  // namespace My
