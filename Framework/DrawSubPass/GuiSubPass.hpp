#pragma once
#include <array>
#include <memory>
#include "BaseSubPass.hpp"

namespace My {
class GuiSubPass : public BaseSubPass {
   public:
    using BaseSubPass::BaseSubPass;
    ~GuiSubPass() override;
    void Draw(Frame& frame) final;

   private:
    void LoadSceneIcons();
    void DrawSceneGraphWindow(const Frame& frame, bool* open);
    void DrawSceneNode(const std::shared_ptr<BaseSceneNode>& node);
    void DrawSceneNodeDetail(const BaseSceneNode& node);

    std::array<Texture2D, 5> m_SceneIcons{};
    bool m_SceneIconsLoaded{false};
    const Scene* m_LastScene{nullptr};
    std::weak_ptr<BaseSceneNode> m_SelectedNode;

    std::vector<Texture2D> m_TextureViews;
    std::array<TextureHandler, 4> m_viewSources{};
    std::array<uint32_t, 3> m_viewSizes{};
};
}  // namespace My