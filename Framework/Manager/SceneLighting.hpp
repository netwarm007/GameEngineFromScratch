#pragma once

#include "cbuffer.h"

namespace My {
class Scene;

struct LightingFrame {
    LightInfo lightInfo{};
    int32_t numLights{0};
    int32_t clipSpaceType{0};
};

// Assign shadow layers before backend uploads; excess casters stay lit without
// shadows and emit a diagnostic.
bool PrepareSceneLights(const Scene& scene, bool openGLProjection,
                        LightingFrame& lighting);
}  // namespace My
