#pragma once
#include <vector>
#include "Scene.hpp"
#include "Asset/Shaders/HLSL/cbuffer.h"

namespace My {
    struct DrawFrameContext {
        Matrix4X4f  m_viewMatrix;
        Matrix4X4f  m_projectionMatrix;
        Vector3f    m_ambientColor;
        Vector3f    m_camPos;
        std::vector<Light> m_lights;
        intptr_t globalShadowMap;
        intptr_t shadowMap;
        intptr_t cubeShadowMap;
        uint32_t globalShadowMapCount;
        uint32_t shadowMapCount;
        uint32_t cubeShadowMapCount;
        intptr_t skybox;

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

    struct DrawBatchContext {
        uint32_t batchIndex;
        std::shared_ptr<SceneGeometryNode> node;
        std::shared_ptr<SceneObjectMaterial> material;
        Matrix4X4f trans;

        virtual ~DrawBatchContext() = default;
    };

    struct Frame {
        DrawFrameContext frameContext;
        std::vector<std::shared_ptr<DrawBatchContext>> batchContexts;
    };
}
