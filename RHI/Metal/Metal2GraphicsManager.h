#pragma once
#include "GraphicsManager.hpp"

#ifdef __OBJC__
#include "MetalView.h"
#include "Metal2Renderer.h"
#endif

namespace My {
    class Metal2GraphicsManager : public GraphicsManager
    {
    public:
       	int Initialize() final;
	    void Finalize() final;

        void Draw() final;
        void Present() final;
    
        void UseShaderProgram(const int32_t shaderProgram) final;
        void SetPerFrameConstants(const DrawFrameContext& context) final;
        void SetPerBatchConstants(const std::vector<std::shared_ptr<DrawBatchContext>>& batches) final;
        void SetLightInfo(const LightInfo& lightInfo) final;

        void DrawBatch(const std::vector<std::shared_ptr<DrawBatchContext>>& batches) final;

        int32_t GenerateCubeShadowMapArray(const uint32_t width, const uint32_t height, const uint32_t count) final;
        int32_t GenerateShadowMapArray(const uint32_t width, const uint32_t height, const uint32_t count) final;
        void BeginShadowMap(const Light& light, const int32_t shadowmap, const uint32_t width, const uint32_t height, const uint32_t layer_index) final;
        void EndShadowMap(const int32_t shadowmap, const uint32_t layer_index) final;
        void SetShadowMaps(const Frame& frame) final;
        void DestroyShadowMap(int32_t& shadowmap) final;

        // skybox
        void SetSkyBox(const DrawFrameContext& context) final;
        void DrawSkyBox() final;

        bool CheckCapability(RHICapability cap) final;
    
#ifdef __OBJC__
        void SetRenderer(Metal2Renderer* renderer) { m_pRenderer = renderer; }
#endif

        void BeginPass() final;
        void EndPass() final;

    private:
        void BeginScene(const Scene& scene) final;
        void EndScene() final;

        void BeginFrame() final;
        void EndFrame() final;

        void initializeGeometries(const Scene& scene);
        void initializeSkyBox(const Scene& scene);
        void initializeTerrain(const Scene& scene);

    private:
#ifdef __OBJC__
        Metal2Renderer* m_pRenderer;
#endif
    };
}
