//
//  GameViewController.m
//  MyGame
//
//  Created by 陈文礼 on 2022/5/10.
//

#import "GameViewController.h"
#import "IApplication.hpp"
#include "InputManager.hpp"
#include "Metal/Metal2GraphicsManager.h"
#include "Metal/Metal2Renderer.h"

@implementation GameViewController
{
    MTKView *_view;

    Metal2Renderer *_renderer;
}

- (instancetype)init {
    _view = (MTKView *)self.view;

    _view.device = MTLCreateSystemDefaultDevice();
    //_view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    //_view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    //_view.framebufferOnly = YES;
    //_view.sampleCount = g_pApp->GetConfiguration().msaaSamples;
    _view.backgroundColor = UIColor.blackColor;
    //_view.paused = YES;
    //_view.enableSetNeedsDisplay = YES;

    if(!_view.device)
    {
        NSLog(@"Metal is not supported on this device");
        self.view = [[UIView alloc] initWithFrame:self.view.frame];
        return;
    }

    _renderer = [[Metal2Renderer alloc] initWithMetalKitView:_view device:_view.device];
    dynamic_cast<Metal2GraphicsManager *>(g_pGraphicsManager)
        ->SetRenderer(_renderer);

    [_renderer mtkView:_view drawableSizeWillChange:_view.bounds.size];

    _view.delegate = _renderer;

    return self;
}

@end
