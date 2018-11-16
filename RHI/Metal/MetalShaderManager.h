#include "ShaderManager.hpp"
#ifdef __OBJC__
#import <MetalKit/MetalKit.h>
#endif

namespace My {
    class MetalShaderManager : public ShaderManager
    {
    public:
        MetalShaderManager() = default;
        ~MetalShaderManager() = default;

        int Initialize() final;
        void Finalize() final;

        void Tick() final;

        bool InitializeShaders() final;
        void ClearShaders() final;

    private:
#ifdef __OBJC__
        id<MTLLibrary> m_shaderLibrary;
#endif
    };
}