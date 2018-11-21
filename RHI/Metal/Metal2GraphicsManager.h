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

        void Clear() final;
        void Draw() final;
        void Present() final;
    
        void UseShaderProgram(const intptr_t shaderProgram) final;
        void SetPerFrameConstants(const DrawFrameContext& context) final;
        void SetPerBatchConstants(const DrawBatchContext& context) final;

        void DrawBatch(const DrawBatchContext& context) final;

        bool CheckCapability(RHICapability cap) final;
    
#ifdef __OBJC__
        void SetRenderer(Metal2Renderer* renderer) { m_pRenderer = renderer; }
#endif

    private:
        void BeginScene(const Scene& scene) final;
        void EndScene() final;

        void BeginFrame() final;
        void EndFrame() final;

    private:
#ifdef __OBJC__
        Metal2Renderer* m_pRenderer;
#endif
    };
}
