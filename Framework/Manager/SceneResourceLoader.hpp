#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "IAssetReader.hpp"
#include "ISceneResourceLoader.hpp"

namespace My {
struct Image;

class SceneResourceLoader : public ISceneResourceLoader {
   public:
    // The reader must support concurrent binary reads during Load().
    explicit SceneResourceLoader(IAssetReader& assets) : m_Assets(assets) {}
    bool Load(Scene& scene) override;

   private:
    std::shared_ptr<Image> LoadImage(const std::string& name);
    std::shared_ptr<Image> GetFallbackImage();

    IAssetReader& m_Assets;
    std::once_flag m_FallbackOnce;
    std::shared_ptr<Image> m_FallbackImage;
};
}  // namespace My
