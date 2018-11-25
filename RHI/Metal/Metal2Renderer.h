#import <MetalKit/MetalKit.h>
#import "GraphicsManager.hpp"

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

- (void)setPerFrameConstants:(const DrawFrameContext&)context;

- (void)setPerBatchConstants:(const std::vector<std::shared_ptr<DrawBatchContext>>&)batches;

- (void)setLightInfo:(const LightInfo&)lightInfo;

- (void)setSkyBox:(const DrawFrameContext&)context;

- (void)drawSkyBox;

- (void)drawBatch:(const std::vector<std::shared_ptr<DrawBatchContext>>&) batches;

- (void)updateDrawableSize:(CGSize)size;

- (void)createVertexBuffer:(const My::SceneObjectVertexArray&)v_property_array;

- (void)createIndexBuffer:(const My::SceneObjectIndexArray&)index_array;

- (uint32_t)createTexture:(const My::Image&)image;

- (uint32_t)createCubeTexture:(const std::vector<const std::shared_ptr<My::Image>>&)images;

- (void)beginFrame;

- (void)endFrame;

@property (nonnull, readonly, nonatomic) id<MTLDevice> device;

@end
