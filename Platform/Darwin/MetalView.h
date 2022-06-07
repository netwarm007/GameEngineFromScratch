#import <MetalKit/MetalKit.h>
#include "IApplication.hpp"

@interface MetalView : MTKView

- (instancetype)initWithFrame:(CGRect)frameRect 
                pApp:(My::IApplication*)pApp;
@end