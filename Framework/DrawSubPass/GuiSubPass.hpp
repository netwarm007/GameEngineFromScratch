#pragma once
#include "BaseSubPass.hpp"

namespace My {
class GuiSubPass : public BaseSubPass {
   public:
    using BaseSubPass::BaseSubPass;
    ~GuiSubPass() override;
    void Draw(Frame& frame) final;

   private:
    std::vector<Texture2D> m_TextureViews;
};
}  // namespace My