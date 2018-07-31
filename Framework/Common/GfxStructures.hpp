#pragma once
#include <vector>
#include "geommath.hpp"
#include "Scene.hpp"

namespace My {
    struct Light{
        Guid        m_lightGuid;
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
        Matrix4X4f  m_worldMatrix;
        Matrix4X4f  m_viewMatrix;
        Matrix4X4f  m_projectionMatrix;
        Vector3f    m_ambientColor;
        std::vector<Light> m_lights;
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
        intptr_t shadowMap;
        uint32_t shadowMapCount;

        Frame ()
        {
            shadowMap = -1;
            shadowMapCount = 0;
        }
    };
}
