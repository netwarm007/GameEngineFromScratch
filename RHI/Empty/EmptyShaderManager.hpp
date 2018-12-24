#include "ShaderManager.hpp"

namespace My {
    class EmptyShaderManager : public ShaderManager
    {
    public:
        EmptyShaderManager() = default;
        ~EmptyShaderManager() = default;

        virtual int Initialize() final { return 0; }
        virtual void Finalize() final {}

        virtual void Tick() final {}

        virtual bool InitializeShaders() final { return true; }
        virtual void ClearShaders() final {}
    };
}