#pragma once
#include <utility>
#include <future>

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
    class SceneObjectTexture : public BaseSceneObject
    {
        private:
            std::string m_Name;
            uint32_t m_nTexCoordIndex;
            std::vector<Matrix4X4f> m_Transforms;
            std::shared_ptr<Image> m_pImage;
            std::future<bool> m_asyncLoadFuture;

        public:
            SceneObjectTexture() : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture) {}
            explicit SceneObjectTexture(std::string  name) : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_Name(std::move(name)), m_nTexCoordIndex(0) 
            { LoadTextureAsync(); }
            SceneObjectTexture(uint32_t coord_index, std::shared_ptr<Image>&& image) : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_nTexCoordIndex(coord_index), m_pImage(std::move(image)) {}

            void AddTransform(Matrix4X4f& matrix) { m_Transforms.push_back(matrix); }
            void SetName(const std::string& name) { m_Name = name; LoadTextureAsync(); }
            void SetName(std::string&& name) { m_Name = std::move(name); LoadTextureAsync(); }
            [[nodiscard]] const std::string& GetName() const { return m_Name; }

            std::shared_ptr<Image> GetTextureImage();

        private:
            bool LoadTexture();
            void LoadTextureAsync();

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectTexture& obj);
    };
}
