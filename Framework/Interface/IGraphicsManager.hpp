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

    virtual intptr_t GenerateCubeShadowMapArray(
        const uint32_t width, const uint32_t height, const uint32_t count) = 0;

    virtual intptr_t GenerateShadowMapArray(
        const uint32_t width, const uint32_t height, const uint32_t count) = 0;

    virtual void BeginShadowMap(
        const int32_t light_index, const intptr_t shadowmap,
        const uint32_t width, const uint32_t height, const int32_t layer_index,
        const Frame& frame) = 0;

    virtual void EndShadowMap(const intptr_t shadowmap,
                              const int32_t layer_index) = 0;

    virtual void SetShadowMaps(const Frame& frame) = 0;

    // skybox
    virtual void DrawSkyBox(const Frame& frame) = 0;

    virtual void GenerateTexture(const char* id, const uint32_t width,
                                 const uint32_t height) = 0;

    virtual void CreateTexture(SceneObjectTexture & texture) = 0;

    virtual void ReleaseTexture(intptr_t texture) = 0;

    virtual void BeginRenderToTexture(int32_t & context, const int32_t texture,
                                      const uint32_t width,
                                      const uint32_t height) = 0;

    virtual void EndRenderToTexture(int32_t & context) = 0;

    virtual void GenerateTextureForWrite(const char* id, const uint32_t width,
                                         const uint32_t height) = 0;

    virtual void BindTextureForWrite(const char* id,
                                     const uint32_t slot_index) = 0;

    virtual void Dispatch(const uint32_t width, const uint32_t height,
                          const uint32_t depth) = 0;

    virtual intptr_t GetTexture(const char* id) = 0;

    virtual void DrawFullScreenQuad() = 0;

#ifdef DEBUG
    virtual void DrawPoint(const Point& point, const Vector3f& color) = 0;
    virtual void DrawPointSet(const PointSet& point_set,
                              const Vector3f& color) = 0;
    virtual void DrawPointSet(const PointSet& point_set,
                              const Matrix4X4f& trans,
                              const Vector3f& color) = 0;
    virtual void DrawLine(const Point& from, const Point& to,
                          const Vector3f& color) = 0;
    virtual void DrawLine(const PointList& vertices, const Vector3f& color) = 0;
    virtual void DrawLine(const PointList& vertices, const Matrix4X4f& trans,
                          const Vector3f& color) = 0;
    virtual void DrawTriangle(const PointList& vertices,
                              const Vector3f& color) = 0;
    virtual void DrawTriangle(const PointList& vertices,
                              const Matrix4X4f& trans,
                              const Vector3f& color) = 0;
    virtual void DrawTriangleStrip(const PointList& vertices,
                                   const Vector3f& color) = 0;
    virtual void DrawTextureOverlay(const intptr_t texture, const float vp_left,
                                    const float vp_top, const float vp_width,
                                    const float vp_height) = 0;
    virtual void DrawTextureArrayOverlay(
        const intptr_t texture, const float layer_index, const float vp_left,
        const float vp_top, const float vp_width, const float vp_height) = 0;
    virtual void DrawCubeMapOverlay(
        const intptr_t cubemap, const float vp_left, const float vp_top,
        const float vp_width, const float vp_height, const float level) = 0;
    virtual void DrawCubeMapArrayOverlay(
        const intptr_t cubemap, const float layer_index, const float vp_left,
        const float vp_top, const float vp_width, const float vp_height,
        const float level) = 0;
    virtual void ClearDebugBuffers() = 0;
#endif
};
}  // namespace My