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

    intptr_t GenerateCubeShadowMapArray(const uint32_t width,
                                        const uint32_t height,
                                        const uint32_t count) override {
        return 0;
    }
    intptr_t GenerateShadowMapArray(const uint32_t width, const uint32_t height,
                                    const uint32_t count) override {
        return 0;
    }
    void BeginShadowMap(const int32_t light_index, const intptr_t shadowmap,
                        const uint32_t width, const uint32_t height,
                        const int32_t layer_index,
                        const Frame& frame) override {}
    void EndShadowMap(const intptr_t shadowmap,
                      const int32_t layer_index) override {}
    void SetShadowMaps(const Frame& frame) override {}

    // skybox
    void DrawSkyBox(const Frame& frame) override {}

    void GenerateTexture(const char* id, const uint32_t width,
                         const uint32_t height) override {}
    void CreateTexture(SceneObjectTexture& texture) override {}
    void ReleaseTexture(intptr_t texture) override {}
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

    intptr_t GetTexture(const char* id) override;

    virtual void DrawFullScreenQuad() override {}

#ifdef DEBUG
    void DrawPoint(const Point& point, const Vector3f& color) override {}
    void DrawPointSet(const PointSet& point_set,
                      const Vector3f& color) override {}
    void DrawPointSet(const PointSet& point_set, const Matrix4X4f& trans,
                      const Vector3f& color) override {}
    void DrawLine(const Point& from, const Point& to,
                  const Vector3f& color) override {}
    void DrawLine(const PointList& vertices, const Vector3f& color) override {}
    void DrawLine(const PointList& vertices, const Matrix4X4f& trans,
                  const Vector3f& color) override {}
    void DrawTriangle(const PointList& vertices,
                      const Vector3f& color) override {}
    void DrawTriangle(const PointList& vertices, const Matrix4X4f& trans,
                      const Vector3f& color) override {}
    void DrawTriangleStrip(const PointList& vertices,
                           const Vector3f& color) override {}
    void DrawTextureOverlay(const intptr_t texture, const float vp_left,
                            const float vp_top, const float vp_width,
                            const float vp_height) override {}
    void DrawTextureArrayOverlay(const intptr_t texture,
                                 const float layer_index, const float vp_left,
                                 const float vp_top, const float vp_width,
                                 const float vp_height) override {}
    void DrawCubeMapOverlay(const intptr_t cubemap, const float vp_left,
                            const float vp_top, const float vp_width,
                            const float vp_height, const float level) override {
    }
    void DrawCubeMapArrayOverlay(const intptr_t cubemap,
                                 const float layer_index, const float vp_left,
                                 const float vp_top, const float vp_width,
                                 const float vp_height,
                                 const float level) override {}
    void ClearDebugBuffers() override {}

    void DrawEdgeList(const EdgeList& edges, const Vector3f& color);
    void DrawPolygon(const Face& polygon, const Vector3f& color);
    void DrawPolygon(const Face& polygon, const Matrix4X4f& trans,
                     const Vector3f& color);
    void DrawPolyhydron(const Polyhedron& polyhedron,
                        const Vector3f& color);
    void DrawPolyhydron(const Polyhedron& polyhedron, const Matrix4X4f& trans,
                        const Vector3f& color);
    void DrawBox(const Vector3f& bbMin, const Vector3f& bbMax,
                 const Vector3f& color);
#endif

   protected:
    virtual void BeginScene(const Scene& scene);
    virtual void EndScene();

    virtual void BeginFrame(const Frame& frame);
    virtual void EndFrame(const Frame& frame);

    virtual void initializeGeometries(const Scene& scene) {}
    virtual void initializeSkyBox(const Scene& scene) {}

#ifdef DEBUG
    virtual void RenderDebugBuffers() {}
#endif

   private:
    void InitConstants() {}
    void CalculateCameraMatrix();
    void CalculateLights();

    void UpdateConstants();

   protected:
    std::map<std::string, intptr_t> m_Textures;

    uint64_t m_nSceneRevision{0};

    uint32_t m_nFrameIndex{0};

    std::array<Frame, GfxConfiguration::kMaxInFlightFrameCount> m_Frames;
    std::vector<std::shared_ptr<IDispatchPass>> m_InitPasses;
    std::vector<std::shared_ptr<IDispatchPass>> m_DispatchPasses;
    std::vector<std::shared_ptr<IDrawPass>> m_DrawPasses;
    
    std::map<std::string, material_textures> material_map;
};
}  // namespace My
