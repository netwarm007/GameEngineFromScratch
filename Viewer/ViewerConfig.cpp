#include "GfxConfiguration.hpp"

#include "ViewerLogic.hpp"
#include "My/MyPhysicsManager.hpp"

namespace My {
GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 1024, 768, "Viewer");
GameLogic* g_pGameLogic = static_cast<GameLogic*>(new ViewerLogic);
PhysicsManager* g_pPhysicsManager =
    static_cast<PhysicsManager*>(new MyPhysicsManager);
}  // namespace My
