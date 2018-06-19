#pragma once
#include <unordered_map>
#include <vector>
#include "geommath.hpp"
#include "Scene.hpp"

namespace My {
    struct Light{
        Guid        m_lightGuid;
        Vector4f    m_lightPosition;
        Vector4f    m_lightColor;
        Vector4f    m_lightDirection;
        float       m_lightIntensity;
        AttenCurveType m_lightDistAttenCurveType;
        float       m_lightDistAttenCurveParams[5];
        AttenCurveType m_lightAngleAttenCurveType;
        float       m_lightAngleAttenCurveParams[5];
        bool        m_bCastShadow;

        Light()
        {
            m_lightPosition = { 0.0f, 0.0f, 0.0f, 1.0f };
            m_lightColor = { 1.0f, 1.0f, 1.0f, 1.0f };
            m_lightDirection = { 0.0f, 0.0f, -1.0f, 0.0f };
            m_lightIntensity = 0.5f;
            m_lightDistAttenCurveType = AttenCurveType::kNone;
            m_lightAngleAttenCurveType = AttenCurveType::kNone;
            m_bCastShadow = false;
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
        std::unordered_map<xg::Guid, intptr_t> shadowMaps;
    };
}
