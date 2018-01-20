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
        virtual int Initialize();
        virtual void Finalize();

        virtual void Clear();

        virtual void Draw();

    protected:
        bool SetPerBatchShaderParameters(const char* paramName, const Matrix4X4f& param);
        bool SetPerBatchShaderParameters(const char* paramName, const Vector3f& param);
        bool SetPerBatchShaderParameters(const char* paramName, const float param);
        bool SetPerBatchShaderParameters(const char* paramName, const int param);
        bool SetPerFrameShaderParameters();

        void InitializeBuffers();
        void RenderBuffers();
        bool InitializeShader(const char* vsFilename, const char* fsFilename);

    private:
        unsigned int m_vertexShader;
        unsigned int m_fragmentShader;
        unsigned int m_shaderProgram;
        std::map<std::string, GLint> m_TextureIndex;

        struct DrawBatchContext {
            GLuint  vao;
            GLenum  mode;
            GLenum  type;
            GLsizei count;
            std::shared_ptr<SceneGeometryNode> node;
            std::shared_ptr<SceneObjectMaterial> material;
        };

        std::vector<DrawBatchContext> m_DrawBatchContext;
        std::vector<GLuint> m_Buffers;
        std::vector<GLuint> m_Textures;
    };

}


