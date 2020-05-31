#pragma once
#include <vector>

#include "GraphicsManager.hpp"
#include "IDrawPass.hpp"
#include "IDrawSubPass.hpp"

namespace My {
class BaseDrawPass : _implements_ IDrawPass {
   public:
    void BeginPass() override { g_pGraphicsManager->BeginPass(); }
    void Draw(Frame& frame) override;
    void EndPass() override { g_pGraphicsManager->EndPass(); }

   protected:
    BaseDrawPass() = default;

   protected:
    std::vector<std::shared_ptr<IDrawSubPass>> m_DrawSubPasses;
};
}  // namespace My
