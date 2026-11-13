#pragma once
#include <array>
#include "BaseSubPass.hpp"

namespace My {
class GuiSubPass : public BaseSubPass {
   public:
    using BaseSubPass::BaseSubPass;
    ~GuiSubPass() override;
    void Draw(Frame& frame) final;

   private:
    std::vector<Texture2D> m_TextureViews;
    std::array<TextureHandler, 4> m_viewSources{};
    std::array<uint32_t, 3> m_viewSizes{};
};
}  // namespace My