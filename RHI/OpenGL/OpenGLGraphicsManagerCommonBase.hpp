#pragma once
#include <unordered_map>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "GraphicsManager.hpp"
#include "geommath.hpp"
#include "SceneManager.hpp"
#include "IApplication.hpp"
#include "IPhysicsManager.hpp"

namespace My {
    class OpenGLGraphicsManagerCommonBase : public GraphicsManager
    {
    public:
        // overrides
        int Initialize() = 0;
        void Finalize() final;

        void Clear() final;

        void Draw() final;

        void UseShaderProgram(const intptr_t shaderProgram) final;
        void SetPerFrameConstants(const DrawFrameContext& context) final;
        void DrawBatch(const DrawBatchContext& context) final;
        void DrawBatchDepthOnly(const DrawBatchContext& context) final;

        intptr_t GenerateCubeShadowMapArray(const uint32_t width, const uint32_t height, const uint32_t count) final;
        intptr_t GenerateShadowMapArray(const uint32_t width, const uint32_t height, const uint32_t count) final;
        void BeginShadowMap(const Light& light, const intptr_t shadowmap, const uint32_t width, const uint32_t height, const uint32_t layer_index) final;
        void EndShadowMap(const intptr_t shadowmap, const uint32_t layer_index) final;
        void SetShadowMaps(const Frame& frame) final;
        void DestroyShadowMap(intptr_t& shadowmap) final;

#ifdef DEBUG
        void DrawPoint(const Point& point, const Vector3f& color) final;
        void DrawPointSet(const PointSet& point_set, const Vector3f& color) final;
        void DrawPointSet(const PointSet& point_set, const Matrix4X4f& trans, const Vector3f& color) final;
        void DrawLine(const Point& from, const Point& to, const Vector3f& color) final;
        void DrawLine(const PointList& vertices, const Vector3f& color) final;
        void DrawLine(const PointList& vertices, const Matrix4X4f& trans, const Vector3f& color) final;
        void DrawTriangle(const PointList& vertices, const Vector3f& color) final;
        void DrawTriangle(const PointList& vertices, const Matrix4X4f& trans, const Vector3f& color) final;
        void DrawTriangleStrip(const PointList& vertices, const Vector3f& color) final;
        void ClearDebugBuffers() final;
        void DrawTextureOverlay(const intptr_t shadowmap, uint32_t layer_index, float vp_left, float vp_top, float vp_width, float vp_height) final;
        void DrawCubeMapOverlay(const intptr_t cubemap, uint32_t layer_index, float vp_left, float vp_top, float vp_width, float vp_height) final;
        void RenderDebugBuffers();
#endif

        void InitializeBuffers(const Scene& scene) final;
        void ClearBuffers() final;

    protected:
        void DrawPoints(const Point* buffer, const size_t count, const Matrix4X4f& trans, const Vector3f& color);

        bool SetShaderParameter(const char* paramName, const Matrix4X4f& param);
        bool SetShaderParameter(const char* paramName, const Matrix4X4f* param, const int32_t count);
        bool SetShaderParameter(const char* paramName, const Vector4f& param);
        bool SetShaderParameter(const char* paramName, const Vector3f& param);
        bool SetShaderParameter(const char* paramName, const Vector2f& param);
        bool SetShaderParameter(const char* paramName, const float param);
        bool SetShaderParameter(const char* paramName, const int32_t param);
        bool SetShaderParameter(const char* paramName, const uint32_t param);
        bool SetShaderParameter(const char* paramName, const bool param);
        bool SetPerFrameShaderParameters(const DrawFrameContext& context);

    private:
        GLuint m_ShadowMapFramebufferName;
        GLuint m_CurrentShader;
        GLuint m_UboBuffer = 0;

        struct OpenGLDrawBatchContext : public DrawBatchContext {
            GLuint  vao;
            GLenum  mode;
            GLenum  type;
            GLsizei count;
        };

#ifdef DEBUG
        struct DebugDrawBatchContext {
            GLuint  vao;
            GLenum  mode;
            GLsizei count;
            Vector3f color;
            Matrix4X4f trans;
        };
#endif

        std::vector<GLuint> m_Buffers;
        std::vector<GLuint> m_Textures;
        std::map<std::string, GLint> m_TextureIndex;

#ifdef DEBUG
        std::vector<DebugDrawBatchContext> m_DebugDrawBatchContext;
        std::vector<GLuint> m_DebugBuffers;
#endif
    };
}
