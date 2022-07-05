#pragma once
#include "BaseSubPass.hpp"

namespace My {
class DebugOverlaySubPass : public BaseSubPass {
   public:
    using BaseSubPass::BaseSubPass;
    ~DebugOverlaySubPass();
    void Draw(Frame& frame) final;

   private:
    std::vector<Texture2D> m_TextureViews;
};
}  // namespace My
