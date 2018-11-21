#import <MetalKit/MetalKit.h>
#import "GraphicsManager.hpp"

namespace My {
    struct MtlDrawBatchContext : public DrawBatchContext {
        uint32_t index_count;
        uint32_t index_offset;
        uint32_t property_count;
        uint32_t property_offset;
    };
}

@interface Metal2Renderer : NSObject

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView
                                      device:(nonnull id <MTLDevice>)device;

- (void)setPerFrameConstants:(const DrawFrameContext&)context;

- (void)setPerBatchConstants:(const DrawBatchContext&)context;

- (void)drawBatch:(const My::MtlDrawBatchContext&)dbc;

- (void)updateDrawableSize:(CGSize)size;

- (void)createVertexBuffer:(const My::SceneObjectVertexArray&)v_property_array;

- (void)createIndexBuffer:(const My::SceneObjectIndexArray&)index_array;

@property (nonnull, readonly, nonatomic) id<MTLDevice> device;

@end
