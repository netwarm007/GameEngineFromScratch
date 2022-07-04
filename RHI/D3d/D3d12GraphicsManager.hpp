#pragma once
#include <DXGI1_4.h>
#include <d3d12.h>
#include <stdint.h>

#include <array>
#include <map>
#include <vector>

#include "Buffer.hpp"
#include "D3d12PipelineStateManager.hpp"
#include "GraphicsManager.hpp"
#include "Image.hpp"
#include "SceneObject.hpp"

namespace My {
class D3d12GraphicsManager : public GraphicsManager {
   public:
    ~D3d12GraphicsManager() override;

    int Initialize() final;
    void Finalize() final;

    void Present() final;

    void SetPipelineState(const std::shared_ptr<PipelineState>& pipelineState,
                          const Frame& frame) final;

    void DrawBatch(const Frame& frame) final;

    void GenerateCubeShadowMapArray(TextureCubeArray& texture_array) final;

    void GenerateShadowMapArray(Texture2DArray& texture_array) final;

    void BeginShadowMap(const int32_t light_index, const TextureID shadowmap,
                        const uint32_t width, const uint32_t height,
                        const int32_t layer_index, const Frame& frame) final;
    void EndShadowMap(const TextureID shadowmap,
                      const int32_t layer_index, const Frame& frame) final;
    void SetShadowMaps(const Frame& frame) final;
    void CreateTexture(SceneObjectTexture& texture) final;
    void ReleaseTexture(TextureID texture) final;

    // skybox
    void DrawSkyBox(const Frame& frame) final;

    void GenerateTextureForWrite(Texture2D& texture) final;

    void BindTextureForWrite(Texture2D& texture, const uint32_t slot_index) final;

    void Dispatch(const uint32_t width, const uint32_t height,
                  const uint32_t depth) final;

   protected:
    void EndScene() final;

    void BeginFrame(const Frame& frame) final;
    void EndFrame(const Frame& frame) final;

    void BeginPass(const Frame& frame) final;
    void EndPass(const Frame& frame) final;

    void BeginCompute() final;
    void EndCompute() final;

    void initializeGeometries(const Scene& scene) final;
    void initializeSkyBox(const Scene& scene) final;

    void SetPerFrameConstants(const Frame& frame);
    void SetLightInfo(const Frame& frame);

    size_t CreateIndexBuffer(const SceneObjectIndexArray& index_array);
    size_t CreateVertexBuffer(const SceneObjectVertexArray& v_property_array);

    HRESULT CreatePSO(D3d12PipelineState& pipelineState);

   private:
    ID3D12DescriptorHeap* m_pCbvSrvUavHeapImGui;

    struct D3dDrawBatchContext : public DrawBatchContext {
        uint32_t index_count{0};
        size_t index_offset{0};
        uint32_t property_count{0};
        size_t property_offset{0};
        size_t cbv_srv_uav_offset{0};
    };

    D3dDrawBatchContext m_dbcSkyBox;
};
}  // namespace My
