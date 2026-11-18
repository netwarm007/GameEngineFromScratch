#include "GfxConfiguration.hpp"

#include "BilliardGameLogic.hpp"
#include "Bullet/BulletPhysicsManager.hpp"

namespace My {
GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 960, 540, "Billiard Game");
GameLogic* CreateBilliardGameLogic(ISceneManager& scenes, IPhysicsManager& physics) {
    return new BilliardGameLogic(scenes, physics);
}
PhysicsManager* CreateBilliardPhysicsManager(ISceneManager& scenes) {
    return new BulletPhysicsManager(scenes);
}
}  // namespace My
