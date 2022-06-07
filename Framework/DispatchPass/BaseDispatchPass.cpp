#include "BaseDispatchPass.hpp"
#include "GraphicsManager.hpp"

using namespace My;

void BaseDispatchPass::BeginPass([[maybe_unused]] Frame& frame) {
    m_pGraphicsManager->BeginCompute(); 
}

void BaseDispatchPass::EndPass([[maybe_unused]] Frame& frame) {
    m_pGraphicsManager->EndCompute(); 
}
