#import <MetalKit/MetalKit.h>

#include "IPipelineStateManager.hpp"

namespace My {
struct MetalPipelineState : public PipelineState {
    MetalPipelineState(PipelineState& rhs) : PipelineState(rhs) {}
    MetalPipelineState(PipelineState&& rhs) : PipelineState(std::move(rhs)) {}

    id<MTLRenderPipelineState> mtlRenderPipelineState;
    id<MTLComputePipelineState> mtlComputePipelineState;
    id<MTLDepthStencilState> depthState;
};
}  // namespace My
