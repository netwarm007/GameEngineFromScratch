#include "GfxConfiguration.hpp"

#include "BilliardGameLogic.hpp"
#include "Bullet/BulletPhysicsManager.hpp"

namespace My {
GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 960, 540, "Billiard Game");
GameLogic* g_pGameLogic = static_cast<GameLogic*>(new BilliardGameLogic);
PhysicsManager* CreateBilliardPhysicsManager(ISceneManager& scenes) {
    return new BulletPhysicsManager(scenes);
}
}  // namespace My
