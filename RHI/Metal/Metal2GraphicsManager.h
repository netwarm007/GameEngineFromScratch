#pragma once
#include "GraphicsManager.hpp"
#include "IPipelineStateManager.hpp"
#include "portable.hpp"

OBJC_CLASS(Metal2Renderer);

namespace My {
class Metal2GraphicsManager : public GraphicsManager {
   public:
    int Initialize() final;
    void Finalize() final;

    void Present() final;

    void SetPipelineState(const std::shared_ptr<PipelineState>& pipelineState,
                          const Frame& frame) final;

    void DrawBatch(const Frame& frame) final;

    intptr_t GenerateCubeShadowMapArray(const uint32_t width,
                                        const uint32_t height,
                                        const uint32_t count) final;
    intptr_t GenerateShadowMapArray(const uint32_t width, const uint32_t height,
                                    const uint32_t count) final;
    void BeginShadowMap(const int32_t light_index, const intptr_t shadowmap,
                        const uint32_t width, const uint32_t height,
                        const int32_t layer_index, const Frame& frame) final;
    void EndShadowMap(const intptr_t shadowmap,
                      const int32_t layer_index) final;
    void SetShadowMaps(const Frame& frame) final;
    void ReleaseTexture(intptr_t texture) final;

    // skybox
    void DrawSkyBox(const Frame& frame) final;

    // compute shader tasks
    void GenerateTextureForWrite(const char* id, const uint32_t width,
                                 const uint32_t height) final;
    void BindTextureForWrite(const char* id, const uint32_t slot_index) final;
    void Dispatch(const uint32_t width, const uint32_t height,
                  const uint32_t depth) final;

#ifdef DEBUG
    void DrawTextureOverlay(const intptr_t texture, const float vp_left,
                            const float vp_top, const float vp_width,
                            const float vp_height) final;

    void DrawTextureArrayOverlay(const intptr_t texture,
                                 const float layer_index, const float vp_left,
                                 const float vp_top, const float vp_width,
                                 const float vp_height) final;

    void DrawCubeMapOverlay(const intptr_t texture, const float vp_left,
                            const float vp_top, const float vp_width,
                            const float vp_height, const float level) final;

    void DrawCubeMapArrayOverlay(const intptr_t texture,
                                 const float layer_index, const float vp_left,
                                 const float vp_top, const float vp_width,
                                 const float vp_height,
                                 const float level) final;
#endif
   protected:
    void BeginFrame(const Frame& frame) final;
    void EndFrame(const Frame& frame) final;

    void BeginPass(const Frame& frame) final;
    void EndPass(const Frame& frame) final;

    void BeginCompute() final;
    void EndCompute() final;

    void initializeGeometries(const Scene& scene) override;
    void initializeSkyBox(const Scene& scene) override;

   private:
    Metal2Renderer* m_pRenderer;
};
}  // namespace My
