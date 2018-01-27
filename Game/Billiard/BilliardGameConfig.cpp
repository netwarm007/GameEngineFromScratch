#include "GfxConfiguration.h"
#include "BilliardGameLogic.hpp"

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 0, 960, 540, "Billiard Game");
    GameLogic*       g_pGameLogic       = static_cast<GameLogic*>(new BilliardGameLogic);
}
