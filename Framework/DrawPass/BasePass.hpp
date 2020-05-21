#pragma once
#include <vector>

#include "GraphicsManager.hpp"
#include "IDrawPass.hpp"
#include "IDrawPhase.hpp"

namespace My {
class BasePass : _implements_ IDrawPass {
   public:
    ~BasePass() override = default;

    void BeginPass() override { g_pGraphicsManager->BeginPass(); }
    void Draw(Frame& frame) override;
    void EndPass() override { g_pGraphicsManager->EndPass(); }

   protected:
    BasePass() = default;

   protected:
    std::vector<std::shared_ptr<IDrawPhase>> m_DrawPhases;
};
}  // namespace My
