#import <MetalKit/MetalKit.h>

@interface Metal2Renderer : NSObject

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView
                                      device:(nonnull id <MTLDevice>)device;

- (void)drawFrameNumber:(NSUInteger)frameNumber toView:(nonnull MTKView *)view;

- (void)updateDrawableSize:(CGSize)size;

@property (nonnull, readonly, nonatomic) id<MTLDevice> device;

@end
