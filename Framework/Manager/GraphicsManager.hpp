#pragma once
#include <array>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "FrameStructure.hpp"
#include "GfxConfiguration.hpp"
#include "IApplication.hpp"
#include "IDispatchPass.hpp"
#include "IDrawPass.hpp"
#include "IGraphicsManager.hpp"
#include "Polyhedron.hpp"
#include "Scene.hpp"
#include "cbuffer.h"
#include "geommath.hpp"

namespace My {
class GraphicsManager : _implements_ IGraphicsManager {
   public:
    GraphicsManager();
    ~GraphicsManager() override = default;
    int Initialize() override;
    void Finalize() override;

    void Tick() override;

    void Draw() override;
    void Present() override {}

    void ResizeCanvas(int32_t width, int32_t height) override;

    void SetPipelineState(const std::shared_ptr<PipelineState>& pipelineState,
                          const Frame& frame) override {}

    void DrawBatch(const Frame& frame) override {}

    void BeginPass(const Frame& frame) override {}
    void EndPass(const Frame& frame) override {}

    void BeginCompute() override {}
    void EndCompute() override {}

    void GenerateTexture(Texture2D& texture) override {}

    void ReleaseTexture(TextureBase& texture) override {}

    void GenerateCubemapArray(TextureCubeArray& texture_array) override {}

    void GenerateTextureArray(Texture2DArray& texture_array) override {}

    void CreateTextureView(Texture2D& texture_view, const TextureArrayBase& texture_array, const uint32_t layer) override {}

    void BeginShadowMap(const int32_t light_index, const TextureBase* pShadowmap,
                        const int32_t layer_index,
                        const Frame& frame) override {}
    void EndShadowMap(const TextureBase* pShadowmap,
                      const int32_t layer_index, const Frame& frame) override {}
    void SetShadowMaps(const Frame& frame) override {}

    // skybox
    void DrawSkyBox(const Frame& frame) override {}

    void CreateTexture(SceneObjectTexture& texture) override {}
    void BeginRenderToTexture(int32_t& context, const int32_t texture,
                              const uint32_t width,
                              const uint32_t height) override {}
    void EndRenderToTexture(int32_t& context) override {}

    void GenerateTextureForWrite(Texture2D& texture) override {}

    void BindTextureForWrite(Texture2D& texture,
                             const uint32_t slot_index) override {}
    void Dispatch(const uint32_t width, const uint32_t height,
                  const uint32_t depth) override {}

    virtual void DrawFullScreenQuad() override {}

   protected:
    virtual void BeginScene(const Scene& scene);
    virtual void EndScene();

    virtual void BeginFrame(const Frame& frame);
    virtual void EndFrame(const Frame& frame);

    virtual void initializeGeometries(const Scene& scene) {}
    virtual void initializeSkyBox(const Scene& scene) {}

   private:
    void InitConstants() {}
    void CalculateCameraMatrix();
    void CalculateLights();

    void UpdateConstants();

   protected:
    uint64_t m_nSceneRevision{0};
    uint32_t m_nFrameIndex{0};

    std::vector<Frame> m_Frames;
    std::vector<std::shared_ptr<IDispatchPass>> m_InitPasses;
    std::vector<std::shared_ptr<IDispatchPass>> m_DispatchPasses;
    std::vector<std::shared_ptr<IDrawPass>> m_DrawPasses;
    
    std::map<std::string, material_textures> material_map;

    std::vector<TextureBase> m_Textures;
    uint32_t m_canvasWidth;
    uint32_t m_canvasHeight;
};
}  // namespace My
