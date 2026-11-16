#import <MetalKit/MetalKit.h>

#include <iostream>

#include "Metal/Metal2GraphicsManager.h"
#include "Metal/Metal2RHI.h"

namespace {
struct AllocationState {
    int failAt = -1;
    int attempts = 0;
    int live = 0;
};
}  // namespace

@interface TrackedMetalResource : NSProxy {
    id _target;
    AllocationState* _state;
}
- (instancetype)initWithOwnedTarget:(id)target state:(AllocationState*)state;
@end

@implementation TrackedMetalResource
- (instancetype)initWithOwnedTarget:(id)target state:(AllocationState*)state {
    _target = target;
    _state = state;
    ++_state->live;
    return self;
}
- (NSMethodSignature*)methodSignatureForSelector:(SEL)selector {
    return [_target methodSignatureForSelector:selector];
}
- (void)forwardInvocation:(NSInvocation*)invocation {
    [invocation invokeWithTarget:_target];
}
- (id<MTLCommandBuffer>)commandBuffer {
    if (_state->attempts++ == _state->failAt) return nil;
    id buffer = [_target commandBuffer];
    if (!buffer) return nil;
    return (id<MTLCommandBuffer>)[[[TrackedMetalResource alloc] initWithOwnedTarget:[buffer retain]
                                                                              state:_state]
        autorelease];
}
- (void)dealloc {
    [_target release];
    --_state->live;
    [super dealloc];
}
@end

@interface FailingMetalDevice : NSProxy {
    id<MTLDevice> _target;
    AllocationState* _state;
}
- (instancetype)initWithDevice:(id<MTLDevice>)device state:(AllocationState*)state;
@end

@implementation FailingMetalDevice
- (instancetype)initWithDevice:(id<MTLDevice>)device state:(AllocationState*)state {
    _target = [device retain];
    _state = state;
    return self;
}
- (NSMethodSignature*)methodSignatureForSelector:(SEL)selector {
    return [(id)_target methodSignatureForSelector:selector];
}
- (void)forwardInvocation:(NSInvocation*)invocation {
    const SEL selector = invocation.selector;
    const bool allocation = selector == @selector(newBufferWithLength:options:) ||
                            selector == @selector(newSamplerStateWithDescriptor:) ||
                            selector == @selector(newCommandQueue);
    if (allocation && _state->attempts++ == _state->failAt) {
        id result = nil;
        [invocation setReturnValue:&result];
        return;
    }
    [invocation invokeWithTarget:_target];
    if (allocation) {
        id result = nil;
        [invocation getReturnValue:&result];
        if (result) {
            result = [[TrackedMetalResource alloc] initWithOwnedTarget:result state:_state];
            [invocation setReturnValue:&result];
        }
    }
}
- (void)dealloc {
    [_target release];
    [super dealloc];
}
@end

int main() {
    @autoreleasepool {
        [NSApplication sharedApplication];
        ImGui::CreateContext();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal failure regression requires a native device\n";
            return 1;
        }
        MTKView* view = [[MTKView alloc] initWithFrame:NSMakeRect(0, 0, 64, 64) device:device];
        const int frames = My::GfxConfiguration::kMaxInFlightFrameCount;
        const int allocations = frames * 3 + 2;
        for (int failAt = 0; failAt < allocations; ++failAt) {
            AllocationState state;
            state.failAt = failAt;
            id proxy = [[FailingMetalDevice alloc] initWithDevice:device state:&state];
            Metal2RHI* rhi = [[Metal2RHI alloc] initWithMetalKitView:view
                                                              device:(id<MTLDevice>)proxy];
            BOOL initialized = NO;
            @autoreleasepool {
                initialized = [rhi initialize];
            }
            if (!rhi || initialized || state.live != 0 || state.attempts != failAt + 1) {
                std::cerr << "Metal partial allocation cleanup failed at " << failAt
                          << ": live=" << state.live << "\n";
                return 1;
            }
            [rhi finalize];
            [rhi finalize];
            state.failAt = -1;
            state.attempts = 0;
            @autoreleasepool {
                initialized = [rhi initialize];
            }
            if (!initialized || [rhi initialize]) {
                std::cerr << "Metal retry/repeated initialization contract failed\n";
                return 1;
            }
            [rhi finalize];
            [rhi finalize];
            [rhi release];
            [proxy release];
            if (state.live != 0) {
                std::cerr << "Metal cleanup leaked tracked native resources\n";
                return 1;
            }
        }
        My::Metal2GraphicsManager missingApplication;
        if (missingApplication.Initialize() == 0) {
            std::cerr << "Missing application must reject Metal initialization\n";
            return 1;
        }
        missingApplication.Finalize();
        missingApplication.Finalize();
        [view release];
        [device release];
        ImGui::DestroyContext();
    }
    return 0;
}
