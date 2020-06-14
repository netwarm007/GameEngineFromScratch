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

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView
                                      device:(nonnull id<MTLDevice>)device;

- (void)setPerFrameConstants:(const DrawFrameContext &)context frameIndex:(const int32_t)index;

- (void)setLightInfo:(const LightInfo &)lightInfo frameIndex:(const int32_t)index;

- (void)drawSkyBox;

- (void)drawBatch:(const Frame &)frame;

- (void)updateDrawableSize:(CGSize)size;

- (void)createVertexBuffer:(const My::SceneObjectVertexArray &)v_property_array;

- (void)createIndexBuffer:(const My::SceneObjectIndexArray &)index_array;

- (uint32_t)createTexture:(const My::Image &)image;

- (uint32_t)createSkyBox:(const std::vector<const std::shared_ptr<My::Image>> &)images;

- (void)beginFrame:(const My::Frame &)frame;

- (void)endFrame:(const My::Frame &)frame;

- (void)beginPass;

- (void)endPass;

- (void)beginCompute;

- (void)endCompute;

- (void)setPipelineState:(const MetalPipelineState &)pipelineState
            frameContext:(const Frame &)frame;

- (int32_t)generateCubeShadowMapArray:(const uint32_t)width
                               height:(const uint32_t)height
                                count:(const uint32_t)count;

- (int32_t)generateShadowMapArray:(const uint32_t)width
                           height:(const uint32_t)height
                            count:(const uint32_t)count;

- (void)beginShadowMap:(const int32_t)light_index
             shadowmap:(const int32_t)shadowmap
                 width:(const uint32_t)width
                height:(const uint32_t)height
           layer_index:(const int32_t)layer_index
                 frame:(const Frame &)frame;

- (void)endShadowMap:(const int32_t)shadowmap layer_index:(const int32_t)layer_index;

- (void)setShadowMaps:(const Frame &)frame;

- (void)releaseTexture:(int32_t)texture;

- (int32_t)generateTextureForWrite:(const uint32_t)width
                            height:(const uint32_t)height;

- (void)bindTextureForWrite:(const uint32_t)id atIndex:(const uint32_t)atIndex;

- (void)dispatch:(const uint32_t)width height:(const uint32_t)height depth:(const uint32_t)depth;

#ifdef DEBUG
- (void)drawTextureOverlay:(const int32_t)texture_id
                   vp_left:(const float)vp_left
                    vp_top:(const float)vp_top
                  vp_width:(const float)vp_width
                 vp_height:(const float)vp_height;

- (void)drawTextureArrayOverlay:(const int32_t)texture_id
                    layer_index:(const float)layer_index
                        vp_left:(const float)vp_left
                         vp_top:(const float)vp_top
                       vp_width:(const float)vp_width
                      vp_height:(const float)vp_height;

- (void)drawCubeMapOverlay:(const int32_t)texture_id
                   vp_left:(const float)vp_left
                    vp_top:(const float)vp_top
                  vp_width:(const float)vp_width
                 vp_height:(const float)vp_height
                     level:(const float)level;

- (void)drawCubeMapArrayOverlay:(const int32_t)texture_id
                    layer_index:(const float)layer_index
                        vp_left:(const float)vp_left
                         vp_top:(const float)vp_top
                       vp_width:(const float)vp_width
                      vp_height:(const float)vp_height
                          level:(const float)level;
#endif

@property(nonnull, readonly, nonatomic) id<MTLDevice> device;

@end
