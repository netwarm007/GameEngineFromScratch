#pragma once
#include <vector>
#include "Scene.hpp"
#include "cbuffer.h"

namespace My {
    struct DrawFrameContext : PerFrameConstants {
        int32_t globalShadowMap;
        int32_t shadowMap;
        int32_t cubeShadowMap;
        int32_t globalShadowMapCount;
        int32_t shadowMapCount;
        int32_t cubeShadowMapCount;
        int32_t skybox;

        DrawFrameContext ()
        {
            globalShadowMap = -1;
            shadowMap = -1;
            cubeShadowMap = -1;
            globalShadowMapCount = 0;
            shadowMapCount = 0;
            cubeShadowMapCount = 0;
        }
    };

    struct DrawBatchContext : PerBatchConstants {
        uint32_t batchIndex;
        std::shared_ptr<SceneGeometryNode> node;
        material_textures material;

        virtual ~DrawBatchContext() = default;
    };

    struct Frame {
        DrawFrameContext frameContext;
        std::vector<std::shared_ptr<DrawBatchContext>> batchContexts;
        LightInfo lightInfo;
    };
}
