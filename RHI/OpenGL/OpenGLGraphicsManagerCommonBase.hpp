#pragma once
#include <string>
#include <vector>

#include "GraphicsManager.hpp"
#include "IApplication.hpp"
#include "IPhysicsManager.hpp"
#include "SceneManager.hpp"
#include "geommath.hpp"

namespace My {
class OpenGLGraphicsManagerCommonBase : public GraphicsManager {
   public:
    // overrides
    void Present() final;

    void SetPipelineState(const std::shared_ptr<PipelineState>& pipelineState,
                          const Frame& frame) final;
    void DrawBatch(const Frame& frame) final;

    void GenerateTexture(Texture2D& texture) final;

    void GenerateCubemapArray(TextureCubeArray& texture_array) final;

    void GenerateTextureArray(Texture2DArray& texture_array) final;

    void CreateTextureView(Texture2D& texture_view,
                           const TextureArrayBase& texture_array,
                           const uint32_t slice,
                           const uint32_t mip) override {}

    void BeginShadowMap(const int32_t light_index,
                        const TextureBase* pShadowmap,
                        const int32_t layer_index, const Frame& frame) final;
    void EndShadowMap(const TextureBase* pShadowmap, const int32_t layer_index,
                      const Frame& frame) final;
    void SetShadowMaps(const Frame& frame) final;
    void ReleaseTexture(TextureBase& texture) final;

    // skybox
    void DrawSkyBox(const Frame& frame) final;

    void GenerateTextureForWrite(Texture2D& texture) final;

    void BindTextureForWrite(Texture2D& texture,
                             const uint32_t slot_index) final;

    void Dispatch(const uint32_t width, const uint32_t height,
                  const uint32_t depth) final;

    void DrawFullScreenQuad() final;

   protected:
    void EndScene() final;

    void BeginFrame(const Frame& frame) override;
    void EndFrame(const Frame& frame) override;

    void initializeGeometries(const Scene& scene) final;
    void initializeSkyBox(const Scene& scene) final;

    void drawPoints(const Point* buffer, const size_t count,
                    const Matrix4X4f& trans, const Vector3f& color);

    void SetPerFrameConstants(const DrawFrameContext& context);
    void SetPerBatchConstants(const DrawBatchContext& context);
    void SetLightInfo(const LightInfo& lightInfo);

    bool setShaderParameter(const char* paramName, const Matrix4X4f& param);
    bool setShaderParameter(const char* paramName, const Matrix4X4f* param,
                            const int32_t count);
    bool setShaderParameter(const char* paramName, const Vector4f& param);
    bool setShaderParameter(const char* paramName, const Vector3f& param);
    bool setShaderParameter(const char* paramName, const Vector2f& param);
    bool setShaderParameter(const char* paramName, const float param);
    bool setShaderParameter(const char* paramName, const int32_t param);
    bool setShaderParameter(const char* paramName, const uint32_t param);
    bool setShaderParameter(const char* paramName, const bool param);

    virtual void getOpenGLTextureFormat(const Image& img, uint32_t& format,
                                        uint32_t& internal_format,
                                        uint32_t& type);

    virtual void getOpenGLTextureFormat(const PIXEL_FORMAT pixel_format,
                                        uint32_t& format,
                                        uint32_t& internal_format,
                                        uint32_t& type) = 0;

    virtual void getOpenGLTextureFormat(const COMPRESSED_FORMAT compressed_format,
                                        uint32_t& format,
                                        uint32_t& internal_format,
                                        uint32_t& type) = 0;

   private:
    uint32_t m_ShadowMapFramebufferName;
    uint32_t m_CurrentShader;
    uint32_t m_uboDrawFrameConstant[GfxConfiguration::kMaxInFlightFrameCount] =
        {0};
    uint32_t m_uboLightInfo[GfxConfiguration::kMaxInFlightFrameCount] = {0};
    uint32_t m_uboDrawBatchConstant[GfxConfiguration::kMaxInFlightFrameCount] =
        {0};
    uint32_t
        m_uboShadowMatricesConstant[GfxConfiguration::kMaxInFlightFrameCount] =
            {0};

    struct OpenGLDrawBatchContext : public DrawBatchContext {
        uint32_t vao{0};
        uint32_t mode{0};
        uint32_t type{0};
        int32_t count{0};
    };

    std::vector<uint32_t> m_Buffers;

    OpenGLDrawBatchContext m_SkyBoxDrawBatchContext;
    OpenGLDrawBatchContext m_TerrainDrawBatchContext;
};
}  // namespace My
