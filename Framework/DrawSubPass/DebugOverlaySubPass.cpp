#include "DebugOverlaySubPass.hpp"

#include "GraphicsManager.hpp"
#include "imgui.h"

using namespace My;
using namespace std;

DebugOverlaySubPass::~DebugOverlaySubPass() {
    for (auto& texture_debug_view : m_TextureViews) {
        m_pGraphicsManager->ReleaseTexture(texture_debug_view);
    }
}

void DebugOverlaySubPass::Draw(Frame& frame) {
}
