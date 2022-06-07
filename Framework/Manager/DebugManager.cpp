#include "DebugManager.hpp"

#include <iostream>

#include "GraphicsManager.hpp"
#include "IGameLogic.hpp"
#include "IPhysicsManager.hpp"

using namespace My;
using namespace std;

#ifdef DEBUG
int DebugManager::Initialize() { return 0; }

void DebugManager::Finalize() {}

void DebugManager::Tick() {
}

void DebugManager::ToggleDebugInfo() { m_bDrawDebugInfo = !m_bDrawDebugInfo; }

#endif