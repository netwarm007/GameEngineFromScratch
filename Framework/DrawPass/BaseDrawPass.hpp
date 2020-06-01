#pragma once
#include <vector>

#include "GraphicsManager.hpp"
#include "IDrawPass.hpp"
#include "IDrawSubPass.hpp"

namespace My {
class BaseDrawPass : _implements_ IDrawPass {
   public:
    ~BaseDrawPass() override = default;

    void BeginPass(Frame& frame) override {
        g_pGraphicsManager->BeginPass(frame);
    }
    void Draw(Frame& frame) override;
    void EndPass(Frame& frame) override { g_pGraphicsManager->EndPass(frame); }

   protected:
    BaseDrawPass() = default;

   protected:
    std::vector<std::shared_ptr<IDrawSubPass>> m_DrawSubPasses;
};
}  // namespace My
