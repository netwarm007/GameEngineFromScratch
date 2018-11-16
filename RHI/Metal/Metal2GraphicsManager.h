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
    
        bool CheckCapability(RHICapability cap) final;
    
#ifdef __OBJC__
        void SetRenderer(Metal2Renderer* renderer) { m_pRenderer = renderer; }
#endif

    protected:
        void BeginScene(const Scene& scene) final;
        void EndScene() final;

    private:
#ifdef __OBJC__
        Metal2Renderer* m_pRenderer;
#endif
    };
}
