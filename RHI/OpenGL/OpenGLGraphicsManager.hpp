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
    public:
        int Initialize();
        void Finalize();

        void Clear();

        void Draw();

#ifdef DEBUG
        void DrawLine(const Vector3f &from, const Vector3f &to, const Vector3f &color);
#endif

    protected:
        bool SetPerBatchShaderParameters(GLuint shader, const char* paramName, const Matrix4X4f& param);
        bool SetPerBatchShaderParameters(GLuint shader, const char* paramName, const Vector3f& param);
        bool SetPerBatchShaderParameters(GLuint shader, const char* paramName, const float param);
        bool SetPerBatchShaderParameters(GLuint shader, const char* paramName, const int param);
        bool SetPerFrameShaderParameters(GLuint shader);

        void InitializeBuffers(const Scene& scene);
        void ClearBuffers();
        bool InitializeShaders();
        void RenderBuffers();

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
        };
#endif

        std::vector<DrawBatchContext> m_DrawBatchContext;
#ifdef DEBUG
        std::vector<DebugDrawBatchContext> m_DebugDrawBatchContext;
#endif

        std::vector<GLuint> m_Buffers;
        std::vector<GLuint> m_Textures;
    };

}


