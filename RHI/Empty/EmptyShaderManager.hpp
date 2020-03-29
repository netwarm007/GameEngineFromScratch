#include "ShaderManager.hpp"

namespace My {
    class EmptyShaderManager : public ShaderManager
    {
    public:
        EmptyShaderManager() = default;
        ~EmptyShaderManager() override = default;

        int Initialize() final { return 0; }
        void Finalize() final {}

        void Tick() final {}

        bool InitializeShaders() final { return true; }
        void ClearShaders() final {}
    };
}