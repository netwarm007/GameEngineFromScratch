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

- (void)beginFrame:(My::Frame &)frame;

- (void)endFrame:(My::Frame &)frame;

- (void)beginPass:(My::Frame &)frame;

- (void)endPass:(My::Frame &)frame;

- (void)drawSkyBox:(const Frame&)frame;

- (void)drawBatch:(const Frame &)frame;

- (void)beginCompute;

- (void)dispatch:(const uint32_t)width height:(const uint32_t)height depth:(const uint32_t)depth;

- (void)endCompute;

- (void)setPipelineState:(const MetalPipelineState &)pipelineState
            frameContext:(const Frame &)frame;

- (_Nonnull id<MTLTexture>)createTexture:(const My::Image &)image;

- (TextureCubeArray)createSkyBox:(const std::vector<const std::shared_ptr<My::Image>> &)images;

- (void)generateCubemapArray:(TextureCubeArray &)texture_array;

- (void)generateTextureArray:(Texture2DArray &)texture_array;

- (void)createTextureView:(Texture2D &)texture_view texture_array:(const TextureArrayBase &)texture_array slice:(const uint32_t)slice mip:(const uint32_t)mip;

- (void)beginShadowMap:(const int32_t)light_index
             shadowmap:(const _Nonnull id<MTLTexture>)shadowmap
                 width:(const uint32_t)width
                height:(const uint32_t)height
           layer_index:(const int32_t)layer_index
                 frame:(const Frame &)frame;

- (void)endShadowMap:(const _Nonnull id<MTLTexture>)shadowmap layer_index:(const int32_t)layer_index frame:(const Frame&)frame;

- (void)setShadowMaps:(const Frame &)frame;

- (void)releaseTexture:(_Nonnull id<MTLTexture>)texture;

- (void)generateTexture:(Texture2D&)texture;

- (void)generateTextureForWrite:(Texture2D &)texture;

- (void)bindTextureForWrite:(const _Nonnull id<MTLTexture>)texture atIndex:(const uint32_t)atIndex;

@property(nonnull, readonly, nonatomic) id<MTLDevice> device;

@end
