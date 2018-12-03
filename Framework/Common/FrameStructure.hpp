#pragma once
#include <vector>
#include "Scene.hpp"
#include "cbuffer.h"

namespace My {
    struct DrawFrameContext : PerFrameConstants, frame_textures {
    };

    struct DrawBatchContext : PerBatchConstants {
        uint32_t batchIndex;
        std::shared_ptr<SceneGeometryNode> node;
        material_textures material;

        virtual ~DrawBatchContext() = default;
    };

    struct Frame : global_textures {
        DrawFrameContext frameContext;
        std::vector<std::shared_ptr<DrawBatchContext>> batchContexts;
        LightInfo lightInfo;
    };
}
