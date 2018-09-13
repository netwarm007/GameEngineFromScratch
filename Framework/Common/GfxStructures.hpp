#pragma once
#include <vector>
#include "geommath.hpp"
#include "Scene.hpp"

namespace My {
    ENUM(LightType) {
        Omni = 0,
        Spot = 1,
        Infinity = 2,
        Area = 3
    };

    struct Light{
        Guid        m_lightGuid;
        LightType   m_lightType;
        Vector4f    m_lightPosition;
        Vector4f    m_lightColor;
        Vector4f    m_lightDirection;
        Vector2f    m_lightSize;
        float       m_lightIntensity;
        AttenCurveType m_lightDistAttenCurveType;
        float       m_lightDistAttenCurveParams[5];
        AttenCurveType m_lightAngleAttenCurveType;
        float       m_lightAngleAttenCurveParams[5];
        bool        m_lightCastShadow;
        int32_t     m_lightShadowMapIndex;
        Matrix4X4f  m_lightVP;

        Light()
        {
            m_lightType = LightType::Omni;
            m_lightPosition = { 0.0f, 0.0f, 0.0f, 1.0f };
            m_lightColor = { 1.0f, 1.0f, 1.0f, 1.0f };
            m_lightDirection = { 0.0f, 0.0f, -1.0f, 0.0f };
            m_lightSize = { 0.0f, 0.0f };
            m_lightIntensity = 0.5f;
            m_lightDistAttenCurveType = AttenCurveType::kNone;
            m_lightAngleAttenCurveType = AttenCurveType::kNone;
            m_lightShadowMapIndex = -1;
        }
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
