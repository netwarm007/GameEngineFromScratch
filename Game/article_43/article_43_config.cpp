#include "GfxConfiguration.h"
#include "article_43_logic.hpp"
#include "My/MyPhysicsManager.hpp"

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 0, 960, 540, "article 43");
    GameLogic*       g_pGameLogic       = static_cast<GameLogic*>(new article_43_logic);
    IPhysicsManager*  g_pPhysicsManager  = static_cast<IPhysicsManager*>(new MyPhysicsManager);
}
