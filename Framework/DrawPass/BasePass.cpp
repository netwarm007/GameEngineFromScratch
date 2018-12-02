#include "BasePass.hpp"

using namespace My;

void BasePass::BeginPass(void)
{ 
    g_pGraphicsManager->BeginPass(); 
}

void BasePass::EndPass(void)
{ 
    g_pGraphicsManager->EndPass(); 
}

void BasePass::Draw(Frame& frame)
{
    for (const auto& pPhase : m_DrawPhases)
    {
        pPhase->BeginPhase();
        pPhase->Draw(frame);
        pPhase->EndPhase();
    }
}