#include "BaseDispatchPass.hpp"
#include "GraphicsManager.hpp"

using namespace My;

void BaseDispatchPass::BeginPass() {
    g_pGraphicsManager->BeginCompute(); 
}

void BaseDispatchPass::EndPass() {
    g_pGraphicsManager->EndCompute(); 
}
