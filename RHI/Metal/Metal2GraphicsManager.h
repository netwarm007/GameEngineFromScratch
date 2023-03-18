#pragma once
#include "GraphicsManager.hpp"
#include "IPipelineStateManager.hpp"
#include "portable.hpp"

OBJC_CLASS(Metal2RHI);

namespace My {
class Metal2GraphicsManager : public GraphicsManager {
   public:
    int Initialize() final;
    void Finalize() final;

    void Present() final;

    void SetPipelineState(const std::shared_ptr<PipelineState>& pipelineState,
                          const Frame& frame) final;

    void DrawBatch(const Frame& frame) final;

    void CreateTextureView(Texture2D& texture_view, const TextureArrayBase& texture_array, const uint32_t slice, const uint32_t mip) final; 

    void GenerateTexture(Texture2D& texture) final;

    void GenerateCubemapArray(TextureCubeArray& texture_array) final;

    void GenerateTextureArray(Texture2DArray& texture_array) final;

    Texture2D CreateTexture(Image& img) final; 

    void BindDebugTexture(Texture2D& texture, const uint32_t slot_index) final;

    void UpdateTexture(Texture2D& texture, Image& img) final;

    void ReleaseTexture(TextureBase& texture) final;

    void BeginShadowMap(const int32_t light_index, const TextureBase* pShadowmap,
                        const int32_t layer_index, const Frame& frame) final;
    void EndShadowMap(const TextureBase* pShadowmap,
                      const int32_t layer_index, const Frame& frame) final;
    void SetShadowMaps(const Frame& frame) final;

    // skybox
    void DrawSkyBox(const Frame& frame) final;

    // compute shader tasks
    void GenerateTextureForWrite(Texture2D& texture) final;
    void BindTextureForWrite(Texture2D& texture, const uint32_t slot_index) final;
    void Dispatch(const uint32_t width, const uint32_t height,
                  const uint32_t depth) final;

   protected:
    void BeginFrame(Frame& frame) final;
    void EndFrame(Frame& frame) final;

    void BeginPass(Frame& frame) final;
    void EndPass(Frame& frame) final;

    void BeginCompute() final;
    void EndCompute() final;

    void initializeGeometries(const Scene& scene) override;
    void initializeSkyBox(const Scene& scene) override;

   private:
    Metal2RHI* m_pRHI;
};
}  // namespace My
