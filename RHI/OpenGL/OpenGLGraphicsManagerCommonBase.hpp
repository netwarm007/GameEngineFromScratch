#pragma once
#include <vector>
#include <unordered_map>
#include <string>
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

        void Present() final;

        void ResizeCanvas(int32_t width, int32_t height) final;

        void UseShaderProgram(const IShaderManager::ShaderHandler shaderProgram) final;
        void DrawBatch(const std::vector<std::shared_ptr<DrawBatchContext>>& batches) final;

        int32_t GenerateCubeShadowMapArray(const uint32_t width, const uint32_t height, const uint32_t count) final;
        int32_t GenerateShadowMapArray(const uint32_t width, const uint32_t height, const uint32_t count) final;
        void BeginShadowMap(const Light& light, const int32_t shadowmap, const uint32_t width, const uint32_t height, const int32_t layer_index) final;
        void EndShadowMap(const int32_t shadowmap, const int32_t layer_index) final;
        void SetShadowMaps(const Frame& frame) final;
        void DestroyShadowMap(int32_t& shadowmap) final;

        // skybox
        void SetSkyBox(const DrawFrameContext& context) final;
        void DrawSkyBox() final;

        // terrain
        void SetTerrain(const DrawFrameContext& context) final;
        void DrawTerrain() final;

        int32_t GenerateTexture(const char* id, const uint32_t width, const uint32_t height) final;
        void BeginRenderToTexture(int32_t& context, const int32_t texture, const uint32_t width, const uint32_t height) final;
        void EndRenderToTexture(int32_t& context) final;
        int32_t GetTexture(const char* id) final;

        int32_t GenerateAndBindTextureForWrite(const char* id, const uint32_t slot_index, const uint32_t width, const uint32_t height) final;
        void Dispatch(const uint32_t width, const uint32_t height, const uint32_t depth) final;

        void DrawFullScreenQuad() final;

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
        void DrawTextureOverlay(const int32_t texture, const float vp_left, const float vp_top, 
                                const float vp_width, const float vp_height) final;
        void DrawTextureArrayOverlay(const int32_t texture, const float layer_index, 
                                     const float vp_left, const float vp_top, const float vp_width, const float vp_height) final;
        void DrawCubeMapOverlay(const int32_t cubemap, const float vp_left, const float vp_top, 
                                const float vp_width, const float vp_height, const float level) final;
        void DrawCubeMapArrayOverlay(const int32_t cubemap, const float layer_index, 
                                     const float vp_left, const float vp_top, const float vp_width, const float vp_height, 
                                     const float level) final;
        void RenderDebugBuffers();
#endif

    private:
        void BeginScene(const Scene& scene) final;
        void EndScene() final;

        void BeginFrame() final;
        void EndFrame() final;

        void BeginPass() final {}
        void EndPass() final {}

        void BeginCompute() final {}
        void EndCompute() final {}

        void initializeGeometries(const Scene& scene);
        void initializeSkyBox(const Scene& scene);
        void initializeTerrain(const Scene& scene);

        void drawPoints(const Point* buffer, const size_t count, const Matrix4X4f& trans, const Vector3f& color);

        void SetPerFrameConstants(const DrawFrameContext& context) final;
        void SetPerBatchConstants(const std::vector<std::shared_ptr<DrawBatchContext>>& batches) final;
        void SetLightInfo(const LightInfo& lightInfo) final;

        bool setShaderParameter(const char* paramName, const Matrix4X4f& param);
        bool setShaderParameter(const char* paramName, const Matrix4X4f* param, const int32_t count);
        bool setShaderParameter(const char* paramName, const Vector4f& param);
        bool setShaderParameter(const char* paramName, const Vector3f& param);
        bool setShaderParameter(const char* paramName, const Vector2f& param);
        bool setShaderParameter(const char* paramName, const float param);
        bool setShaderParameter(const char* paramName, const int32_t param);
        bool setShaderParameter(const char* paramName, const uint32_t param);
        bool setShaderParameter(const char* paramName, const bool param);

        virtual void getOpenGLTextureFormat(const Image& img, uint32_t& format, uint32_t& internal_format, uint32_t& type) = 0;

    private:
        uint32_t m_ShadowMapFramebufferName;
        uint32_t m_CurrentShader;
        uint32_t m_uboDrawFrameConstant[GfxConfiguration::kMaxInFlightFrameCount] = {0};
        uint32_t m_uboLightInfo[GfxConfiguration::kMaxInFlightFrameCount] = {0};
        uint32_t m_uboDrawBatchConstant[GfxConfiguration::kMaxInFlightFrameCount] = {0};
        uint32_t m_uboShadowMatricesConstant[GfxConfiguration::kMaxInFlightFrameCount] = {0};
        uint32_t m_uboDebugConstant[GfxConfiguration::kMaxInFlightFrameCount] = {0};

        struct OpenGLDrawBatchContext : public DrawBatchContext {
            uint32_t vao;
            uint32_t mode;
            uint32_t type;
            int32_t count;
        };

#ifdef DEBUG
        struct DebugDrawBatchContext : public OpenGLDrawBatchContext {
            Vector3f color;
            Matrix4X4f trans;
        };
#endif

        std::vector<uint32_t> m_Buffers;
        std::unordered_map<std::string, uint32_t> m_Textures;

#ifdef DEBUG
        std::vector<DebugDrawBatchContext> m_DebugDrawBatchContext;
        std::vector<uint32_t> m_DebugBuffers;
#endif

        OpenGLDrawBatchContext m_SkyBoxDrawBatchContext;
        OpenGLDrawBatchContext m_TerrainDrawBatchContext;
    };
}
