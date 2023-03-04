#import <Cocoa/Cocoa.h>
#include "IApplication.hpp"

@interface WindowDelegate : NSObject <NSWindowDelegate>

- (instancetype)initWithApp:(My::IApplication*)pApp;

@end
