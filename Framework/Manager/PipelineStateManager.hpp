#include <map>
#include "IApplication.hpp"
#include "IPipelineStateManager.hpp"

namespace My {
class PipelineStateManager : _implements_ IPipelineStateManager {
   public:
    ~PipelineStateManager() override;

    int Initialize() override;
    void Finalize() override;
    void Tick() override {}

    bool RegisterPipelineState(PipelineState& pipelineState) override;
    void UnregisterPipelineState(PipelineState& pipelineState) override;
    void Clear() override;

    const std::shared_ptr<PipelineState> GetPipelineState(
        std::string name) const final;

   protected:
    virtual bool InitializePipelineState(PipelineState** ppPipelineState) {
        return true;
    }
    virtual void DestroyPipelineState(PipelineState& pipelineState) {}

   protected:
    std::map<std::string, std::shared_ptr<PipelineState>> m_pipelineStates;
};
}  // namespace My
