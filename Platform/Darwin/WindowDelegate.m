#import "WindowDelegate.h"

@interface WindowDelegate ()

@end

@implementation WindowDelegate

- (void)windowWillClose:(NSNotification *)wNotification {
    [NSApp terminate:self];
}

@end
