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

    TextureID GenerateCubeShadowMapArray(const uint32_t width,
                                        const uint32_t height,
                                        const uint32_t count) override {
        return 0;
    }
    TextureID GenerateShadowMapArray(const uint32_t width, const uint32_t height,
                                    const uint32_t count) override {
        return 0;
    }
    void BeginShadowMap(const int32_t light_index, const TextureID shadowmap,
                        const uint32_t width, const uint32_t height,
                        const int32_t layer_index,
                        const Frame& frame) override {}
    void EndShadowMap(const TextureID shadowmap,
                      const int32_t layer_index, const Frame& frame) override {}
    void SetShadowMaps(const Frame& frame) override {}

    // skybox
    void DrawSkyBox(const Frame& frame) override {}

    void GenerateTexture(const char* id, const uint32_t width,
                         const uint32_t height) override {}
    void CreateTexture(SceneObjectTexture& texture) override {}
    void ReleaseTexture(TextureID texture) override {}
    void BeginRenderToTexture(int32_t& context, const int32_t texture,
                              const uint32_t width,
                              const uint32_t height) override {}
    void EndRenderToTexture(int32_t& context) override {}

    void GenerateTextureForWrite(const char* id, const uint32_t width,
                                 const uint32_t height) override {}

    void BindTextureForWrite(const char* id,
                             const uint32_t slot_index) override {}
    void Dispatch(const uint32_t width, const uint32_t height,
                  const uint32_t depth) override {}

    TextureID GetTexture(const char* id) override;

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
    std::map<std::string, TextureID> m_Textures;

    uint64_t m_nSceneRevision{0};
    uint32_t m_nFrameIndex{0};

    std::array<Frame, GfxConfiguration::kMaxInFlightFrameCount> m_Frames;
    std::vector<std::shared_ptr<IDispatchPass>> m_InitPasses;
    std::vector<std::shared_ptr<IDispatchPass>> m_DispatchPasses;
    std::vector<std::shared_ptr<IDrawPass>> m_DrawPasses;
    
    std::map<std::string, material_textures> material_map;
};
}  // namespace My
