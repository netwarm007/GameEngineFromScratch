#pragma once
#include "IRuntimeModule.hpp"

#include <memory>
#include "FrameStructure.hpp"
#include "IPipelineStateManager.hpp"

namespace My {
_Interface_ IGraphicsManager : _inherits_ IRuntimeModule {
   public:
    IGraphicsManager() = default;
    virtual ~IGraphicsManager() = default;
    virtual void Draw() = 0;
    virtual void Present() = 0;

    virtual void ResizeCanvas(int32_t width, int32_t height) = 0;

    virtual void SetPipelineState(
        const std::shared_ptr<PipelineState>& pipelineState,
        const Frame& frame) = 0;

    virtual void DrawBatch(const Frame& frame) = 0;

    virtual void BeginPass(const Frame& frame) = 0;
    virtual void EndPass(const Frame& frame) = 0;

    virtual void BeginCompute() = 0;
    virtual void EndCompute() = 0;

    virtual void CreateTexture(SceneObjectTexture & texture) = 0;

    virtual void CreateTextureView(Texture2D& texture_view, const TextureArrayBase& texture_array, const uint32_t slice, const uint32_t mip) = 0;

    virtual void GenerateTexture(Texture2D& texture) = 0;

    virtual void GenerateCubemapArray(TextureCubeArray& texture_array) = 0;

    virtual void GenerateTextureArray(Texture2DArray& texture_array) = 0;

    virtual void ReleaseTexture(TextureBase& texture) = 0;

    virtual void BeginShadowMap(
        const int32_t light_index, const TextureBase* pShadowmap,
        const int32_t layer_index,
        const Frame& frame) = 0;

    virtual void EndShadowMap(const TextureBase* pShadowmap,
                              const int32_t layer_index, const Frame& frame) = 0;

    virtual void SetShadowMaps(const Frame& frame) = 0;

    // skybox
    virtual void DrawSkyBox(const Frame& frame) = 0;

    virtual void GenerateTextureForWrite(Texture2D& texture) = 0;

    virtual void BindTextureForWrite(Texture2D& texture,
                                     const uint32_t slot_index) = 0;

    virtual void Dispatch(const uint32_t width, const uint32_t height,
                          const uint32_t depth) = 0;

    virtual void DrawFullScreenQuad() = 0;
};
}  // namespace My