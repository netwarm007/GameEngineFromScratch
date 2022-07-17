#import "WindowDelegate.h"

@interface WindowDelegate ()

@end

@implementation WindowDelegate {
    My::IApplication* m_pApp;
}

- (instancetype)initWithApp:(My::IApplication *)pApp {
    m_pApp = pApp;

    return self;
}

- (void)windowWillClose:(NSNotification *)wNotification {
    assert(m_pApp && "m_pApp must not be NULL!");
    m_pApp->RequestQuit();
}

@end
