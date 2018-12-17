#include "BasePass.hpp"

using namespace My;

void BasePass::Draw(Frame& frame)
{
    for (const auto& pPhase : m_DrawPhases)
    {
        pPhase->BeginPhase();
        pPhase->Draw(frame);
        pPhase->EndPhase();
    }
}