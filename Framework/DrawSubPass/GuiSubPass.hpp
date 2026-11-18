#pragma once
#include <array>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include "BaseSubPass.hpp"

namespace My {
class GuiSubPass : public BaseSubPass {
   public:
    using IconImageLoader = std::function<Image(const char*)>;
    GuiSubPass(IGraphicsManager* graphics, IPipelineStateManager* pipelines,
               IconImageLoader loadIconImage)
        : BaseSubPass(graphics, pipelines),
          m_LoadIconImage(std::move(loadIconImage)) {
        if (!m_LoadIconImage) {
            throw std::invalid_argument("GuiSubPass requires an icon image loader");
        }
    }
    ~GuiSubPass() override;
    void Draw(Frame& frame) final;

   private:
    void LoadSceneIcons();
    void DrawSceneGraphWindow(const Frame& frame, bool* open);
    void DrawSceneNode(const std::shared_ptr<BaseSceneNode>& node);
    void DrawSceneNodeDetail(const BaseSceneNode& node);

    std::array<Texture2D, 5> m_SceneIcons{};
    IconImageLoader m_LoadIconImage;
    bool m_SceneIconsLoaded{false};
    const Scene* m_LastScene{nullptr};
    std::weak_ptr<BaseSceneNode> m_SelectedNode;
    bool m_DockLayoutInitialized{false};
    bool m_ResetDockLayout{false};

    std::vector<Texture2D> m_TextureViews;
    std::weak_ptr<const Scene> m_ViewScene;
    uint64_t m_ViewResourceRevision{0};
    std::array<TextureHandler, 4> m_viewSources{};
    std::array<uint32_t, 3> m_viewSizes{};
};
}  // namespace My