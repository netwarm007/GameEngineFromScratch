#include "PipelineStateManager.hpp"

namespace My {
class EmptyPipelineStateManager : public PipelineStateManager {
   public:
    EmptyPipelineStateManager() = default;
    ~EmptyPipelineStateManager() = default;

    int Initialize() final { return 0; }
    void Finalize() final {}

    void Tick() final {}

   protected:
    bool InitializePipelineState(PipelineState** ppPipelineState) final;
    void DestroyPipelineState(PipelineState& pipelineState) final;
};
}  // namespace My