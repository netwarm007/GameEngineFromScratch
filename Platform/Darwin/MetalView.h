#import <MetalKit/MetalKit.h>
#include "IInputManager.hpp"

@interface MetalView : MTKView

- (instancetype)initWithFrame:(CGRect)frameRect
                        input:(My::IInputManager*)inputManager
                  msaaSamples:(uint32_t)msaaSamples;
@end