#pragma once
#include <vector>
#include <memory>
#include "IRuntimeModule.hpp"
#include "IShaderManager.hpp"
#include "geommath.hpp"
#include "Image.hpp"
#include "Scene.hpp"
#include "Polyhedron.hpp"

namespace My {
    class GraphicsManager : implements IRuntimeModule
    {
    public:
        virtual ~GraphicsManager() {}

        virtual int Initialize();
        virtual void Finalize();

        virtual void Tick();

        virtual void Clear();
        virtual void Draw();

#ifdef DEBUG
        virtual void DrawPoint(const Point& point, const Vector3f& color);
        virtual void DrawPointSet(const PointSet& point_set, const Vector3f& color);
        virtual void DrawPointSet(const PointSet& point_set, const Matrix4X4f& trans, const Vector3f& color);
        virtual void DrawLine(const Point& from, const Point& to, const Vector3f &color);
        virtual void DrawLine(const PointList& vertices, const Vector3f &color);
        virtual void DrawLine(const PointList& vertices, const Matrix4X4f& trans, const Vector3f &color);
        virtual void DrawTriangle(const PointList& vertices, const Vector3f &color);
        virtual void DrawTriangle(const PointList& vertices, const Matrix4X4f& trans, const Vector3f &color);
        virtual void DrawTriangleStrip(const PointList& vertices, const Vector3f &color);
        void DrawEdgeList(const EdgeList& edges, const Vector3f& color);
        void DrawPolygon(const Face& face, const Vector3f& color);
        void DrawPolygon(const Face& face, const Matrix4X4f& trans, const Vector3f& color);
        void DrawPolyhydron(const Polyhedron& polyhedron, const Vector3f& color);
        void DrawPolyhydron(const Polyhedron& polyhedron, const Matrix4X4f& trans, const Vector3f& color);
        void DrawBox(const Vector3f& bbMin, const Vector3f& bbMax, const Vector3f& color);
        virtual void ClearDebugBuffers();
#endif

    protected:
        virtual void InitializeBuffers(const Scene& scene);
        virtual void ClearBuffers();

        virtual void InitConstants();
        virtual void CalculateCameraMatrix();
        virtual void CalculateLights();
        virtual void UpdateConstants();
        virtual void RenderBuffers();
#ifdef DEBUG
        virtual void RenderDebugBuffers();
#endif

    protected:
        struct Light {
            Vector4f    m_lightPosition;
            Vector4f    m_lightColor;
            Vector4f    m_lightDirection;
            Vector2f    m_lightSize;
            float       m_lightIntensity;
            AttenCurveType m_lightDistAttenCurveType;
            float       m_lightDistAttenCurveParams[5];
            AttenCurveType m_lightAngleAttenCurveType;
            float       m_lightAngleAttenCurveParams[5];

            Light()
            {
                m_lightPosition = { 0.0f, 0.0f, 0.0f, 1.0f };
                m_lightSize = { 0.0f, 0.0f };
                m_lightColor = { 1.0f, 1.0f, 1.0f, 1.0f };
                m_lightDirection = { 0.0f, 0.0f, -1.0f, 0.0f };
                m_lightIntensity = 0.5f;
                m_lightDistAttenCurveType = AttenCurveType::kNone;
                m_lightAngleAttenCurveType = AttenCurveType::kNone;
            }
        };

        struct DrawFrameContext {
            Matrix4X4f  m_worldMatrix;
            Matrix4X4f  m_viewMatrix;
            Matrix4X4f  m_projectionMatrix;
            Vector3f    m_ambientColor;
            std::vector<Light> m_lights;
        };

        DrawFrameContext    m_DrawFrameContext;
    };

    extern GraphicsManager* g_pGraphicsManager;
}

