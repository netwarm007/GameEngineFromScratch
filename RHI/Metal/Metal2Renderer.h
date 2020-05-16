#import "GraphicsManager.hpp"
#import <MetalKit/MetalKit.h>
#import "MetalPipelineStateManager.h"
#import "MetalPipelineState.h"

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

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView
                                      device:(nonnull id <MTLDevice>)device;

- (void)setPerFrameConstants:(const DrawFrameContext&)context
          frameIndex:(const int32_t)index;

- (void)setLightInfo:(const LightInfo&)lightInfo
          frameIndex:(const int32_t)index;

- (void)drawSkyBox;

- (void)drawBatch:(const Frame&) frame;

- (void)updateDrawableSize:(CGSize)size;

- (void)createVertexBuffer:(const My::SceneObjectVertexArray&)v_property_array;

- (void)createIndexBuffer:(const My::SceneObjectIndexArray&)index_array;

- (uint32_t)createTexture:(const My::Image&)image;

- (uint32_t)createSkyBox:(const std::vector<const std::shared_ptr<My::Image>>&)images;

- (void)beginFrame:(const My::Frame&)frame;

- (void)endFrame;

- (void)beginPass;

- (void)endPass;

- (void)beginCompute;

- (void)endCompute;

- (void)setPipelineState:(const MetalPipelineState&)pipelineState frameContext:(const Frame&)frame;

- (int32_t)generateCubeShadowMapArray:(const uint32_t)width 
                               height:(const uint32_t)height
                                count:(const uint32_t)count;

- (int32_t)generateShadowMapArray:(const uint32_t)width
                           height:(const uint32_t)height
                            count:(const uint32_t)count;

- (void)beginShadowMap:(const Light&)light
             shadowmap:(const int32_t)shadowmap
                 width:(const uint32_t)width
                height:(const uint32_t)height
           layer_index:(const int32_t)layer_index
                 frame:(const Frame&)frame;

- (void)endShadowMap:(const int32_t)shadowmap
         layer_index:(const int32_t)layer_index;

- (void)setShadowMaps:(const Frame&)frame;

- (void)destroyShadowMap:(int32_t&)shadowmap;

- (int32_t)generateAndBindTextureForWrite:(const uint32_t)width
                                   height:(const uint32_t)height
                                  atIndex:(const uint32_t)atIndex;

- (void)dispatch:(const uint32_t)width
          height:(const uint32_t)height
           depth:(const uint32_t)depth;

@property (nonnull, readonly, nonatomic) id<MTLDevice> device;

@end
