#pragma once
#include <vector>
#include "geommath.hpp"
#include "Scene.hpp"

namespace My {
    typedef enum LightType {
        Omni = 0,
        Spot = 1,
        Infinity = 2,
        Area = 3
    } LightType;

    struct Light{
        Guid        m_lightGuid;
        LightType   m_lightType = LightType::Omni;
        Vector4f    m_lightPosition = { 0.0f, 0.0f, 0.0f, 1.0f };
        Vector4f    m_lightColor = { 1.0f, 1.0f, 1.0f, 1.0f };
        Vector4f    m_lightDirection = { 0.0f, 0.0f -1.0f, 0.0f };
        Vector4f    m_lightSize = { 0.0f, 0.0f };
        float       m_lightIntensity = 1.0f;
        AttenCurveType m_lightDistAttenCurveType = AttenCurveType::kNone;
        float       m_lightDistAttenCurveParams[5];
        AttenCurveType m_lightAngleAttenCurveType = AttenCurveType::kNone;
        float       m_lightAngleAttenCurveParams[5];
        bool        m_lightCastShadow;
        int32_t     m_lightShadowMapIndex = -1;
        Matrix4X4f  m_lightVP;
    };

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
