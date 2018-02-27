#include "GfxConfiguration.h"
#include "article_44_logic.hpp"
#include "My/MyPhysicsManager.hpp"

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 0, 960, 540, "article 44");
    IGameLogic*       g_pGameLogic       = static_cast<IGameLogic*>(new article_44_logic);
    IPhysicsManager*  g_pPhysicsManager  = static_cast<IPhysicsManager*>(new MyPhysicsManager);
}
