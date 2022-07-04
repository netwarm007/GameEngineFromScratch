#pragma once
#import <MetalKit/MetalKit.h>
#import "GraphicsManager.hpp"
#import "MetalPipelineState.h"
#import "MetalPipelineStateManager.h"

namespace My {
struct MtlDrawBatchContext : public DrawBatchContext {
    uint32_t index_count;
    uint32_t index_offset;
    MTLPrimitiveType index_mode;
    MTLIndexType index_type;
    uint32_t property_count;
    uint32_t property_offset;
};
}

@interface Metal2Renderer : NSObject

- (void)initialize;

- (void)finalize;

- (void)present;

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView
                                      device:(nonnull id<MTLDevice>)device;

- (void)setPerFrameConstants:(const DrawFrameContext &)context frameIndex:(const int32_t)index;

- (void)setLightInfo:(const LightInfo &)lightInfo frameIndex:(const int32_t)index;

- (void)updateDrawableSize:(CGSize)size;

- (void)createVertexBuffer:(const My::SceneObjectVertexArray &)v_property_array;

- (void)createIndexBuffer:(const My::SceneObjectIndexArray &)index_array;

- (void)beginFrame:(const My::Frame &)frame;

- (void)endFrame:(const My::Frame &)frame;

- (void)beginPass:(const My::Frame &)frame;

- (void)endPass:(const My::Frame &)frame;

- (void)drawSkyBox:(const Frame&)frame;

- (void)drawBatch:(const Frame &)frame;

- (void)beginCompute;

- (void)dispatch:(const uint32_t)width height:(const uint32_t)height depth:(const uint32_t)depth;

- (void)endCompute;

- (void)setPipelineState:(const MetalPipelineState &)pipelineState
            frameContext:(const Frame &)frame;

- (_Nonnull id<MTLTexture>)createTexture:(const My::Image &)image;

- (_Nonnull id<MTLTexture>)createSkyBox:(const std::vector<const std::shared_ptr<My::Image>> &)images;

- (_Nonnull id<MTLTexture>)generateCubeShadowMapArray:(const uint32_t)width
                               height:(const uint32_t)height
                                count:(const uint32_t)count;

- (_Nonnull id<MTLTexture>)generateShadowMapArray:(const uint32_t)width
                           height:(const uint32_t)height
                            count:(const uint32_t)count;

- (void)beginShadowMap:(const int32_t)light_index
             shadowmap:(const _Nonnull id<MTLTexture>)shadowmap
                 width:(const uint32_t)width
                height:(const uint32_t)height
           layer_index:(const int32_t)layer_index
                 frame:(const Frame &)frame;

- (void)endShadowMap:(const _Nonnull id<MTLTexture>)shadowmap layer_index:(const int32_t)layer_index frame:(const Frame&)frame;

- (void)setShadowMaps:(const Frame &)frame;

- (void)releaseTexture:(_Nonnull id<MTLTexture>)texture;

- (_Nonnull id<MTLTexture>)generateTextureForWrite:(const uint32_t)width
                            height:(const uint32_t)height;

- (void)bindTextureForWrite:(const _Nonnull id<MTLTexture>)texture atIndex:(const uint32_t)atIndex;

@property(nonnull, readonly, nonatomic) id<MTLDevice> device;

@end
