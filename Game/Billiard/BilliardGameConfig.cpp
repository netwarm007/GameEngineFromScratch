#include "GfxConfiguration.hpp"

#include "BilliardGameLogic.hpp"
#include "Bullet/BulletPhysicsManager.hpp"

namespace My {
GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 960, 540, "Billiard Game");
GameLogic* g_pGameLogic = static_cast<GameLogic*>(new BilliardGameLogic);
PhysicsManager* g_pPhysicsManager =
    static_cast<PhysicsManager*>(new BulletPhysicsManager);
}  // namespace My
