#import <MetalKit/MetalKit.h>

#include "IPipelineStateManager.hpp"

namespace My {
struct MetalPipelineState : public PipelineState {
    MetalPipelineState(PipelineState& rhs) : PipelineState(rhs) {}
    MetalPipelineState(PipelineState&& rhs) : PipelineState(std::move(rhs)) {}

    id<MTLRenderPipelineState> mtlRenderPipelineState = Nil;
    id<MTLComputePipelineState> mtlComputePipelineState = Nil;
    id<MTLDepthStencilState> depthState = Nil;
};
}  // namespace My
