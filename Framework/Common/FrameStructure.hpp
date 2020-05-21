#pragma once
#include <vector>

#include "Scene.hpp"
#include "cbuffer.h"

namespace My {
struct DrawFrameContext : PerFrameConstants, frame_textures {};

struct DrawBatchContext : PerBatchConstants {
    int32_t batchIndex{0};
    std::shared_ptr<SceneGeometryNode> node;
    material_textures material;

    virtual ~DrawBatchContext() = default;
};

struct Frame : global_textures {
    int32_t frameIndex{0};
    DrawFrameContext frameContext;
    std::vector<std::shared_ptr<DrawBatchContext>> batchContexts;
    LightInfo lightInfo;
};
}  // namespace My
