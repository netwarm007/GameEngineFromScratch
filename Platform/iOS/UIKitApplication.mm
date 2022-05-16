#include "UIKitApplication.h"
#include <cstring>
#import "AppDelegate.h"
#include "InputManager.hpp"
#include "imgui_impl_ios.h"

#import "UIKit/UIKit.h"
#import "Foundation/Foundation.h"

#import "GameViewController.h"

using namespace My;

void* UIKitApplication::GetMainWindowHandler() { return m_pWindow; }

void UIKitApplication::CreateMainWindow() {
    [UIApplication sharedApplication].statusBarHidden = YES;

    m_pWindow = [[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]];
    m_pWindow.rootViewController = [[GameViewController new] init];
    [m_pWindow makeKeyAndVisible];

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    [[maybe_unused]] ImGuiIO& io = ImGui::GetIO();

    ImGui_ImplIOS_Init();

    ImGui::StyleColorsDark();
}

void UIKitApplication::Finalize() {
    ImGui_ImplIOS_Shutdown();

    [m_pWindow release];

    BaseApplication::Finalize();
}

void UIKitApplication::Tick() {
}
