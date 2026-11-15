#pragma once
#include <utility>

#include "BaseSceneObject.hpp"
#include "geommath.hpp"
#include "Image.hpp"

namespace My {
class SceneObjectTexture : public BaseSceneObject {
   private:
    std::string m_Name;
    uint32_t m_nTexCoordIndex{0};
    std::vector<Matrix4X4f> m_Transforms;
    std::shared_ptr<Image> m_pImage;

   public:
    SceneObjectTexture()
        : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture) {}
    explicit SceneObjectTexture(const std::string& name)
        : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture),
          m_Name(name) {}

    void AddTransform(Matrix4X4f& matrix) { m_Transforms.push_back(matrix); }
    void SetName(const std::string& name) {
        m_Name = name;
        m_pImage.reset();
    }
    void SetName(std::string&& name) {
        m_Name = std::move(name);
        m_pImage.reset();
    }
    [[nodiscard]] const std::string& GetName() const { return m_Name; }

    std::shared_ptr<Image> GetTextureImage() const { return m_pImage; }
    void SetTextureImage(std::shared_ptr<Image> image) {
        m_pImage = std::move(image);
    }

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectTexture& obj);
};
}  // namespace My
