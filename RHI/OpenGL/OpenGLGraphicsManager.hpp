#pragma once
#include <unordered_map>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "GraphicsManager.hpp"
#include "geommath.hpp"
#include "glad/glad.h"
#include "SceneManager.hpp"

namespace My {
    class OpenGLGraphicsManager : public GraphicsManager
    {
        // overrides
        int Initialize() final;
        void Finalize() final;

        void Clear() final;

        void Draw() final;

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
        void RenderDebugBuffers();
#endif

        void InitializeBuffers(const Scene& scene) final;
        void ClearBuffers() final;
        bool InitializeShaders() final;
        void ClearShaders() final;
        void RenderBuffers() final;

    protected:
        void DrawPoints(const Point* buffer, const size_t count, const Matrix4X4f& trans, const Vector3f& color);

        bool SetPerBatchShaderParameters(GLuint shader, const char* paramName, const Matrix4X4f& param);
        bool SetPerBatchShaderParameters(GLuint shader, const char* paramName, const Vector3f& param);
        bool SetPerBatchShaderParameters(GLuint shader, const char* paramName, const float param);
        bool SetPerBatchShaderParameters(GLuint shader, const char* paramName, const int param);
        bool SetPerFrameShaderParameters(GLuint shader);

    private:
        GLuint m_vertexShader;
        GLuint m_fragmentShader;
        GLuint m_shaderProgram;
#ifdef DEBUG
        GLuint m_debugVertexShader;
        GLuint m_debugFragmentShader;
        GLuint m_debugShaderProgram;
#endif
        std::map<std::string, GLint> m_TextureIndex;

        struct DrawBatchContext {
            GLuint  vao;
            GLenum  mode;
            GLenum  type;
            GLsizei count;
            std::shared_ptr<SceneGeometryNode> node;
            std::shared_ptr<SceneObjectMaterial> material;
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

        std::vector<DrawBatchContext> m_DrawBatchContext;

        std::vector<GLuint> m_Buffers;
        std::vector<GLuint> m_Textures;

#ifdef DEBUG
        std::vector<DebugDrawBatchContext> m_DebugDrawBatchContext;
        std::vector<GLuint> m_DebugBuffers;
#endif
    };

}


