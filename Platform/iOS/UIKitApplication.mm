#include "UIKitApplication.h"
#include <cstring>
#import "AppDelegate.h"
#include "InputManager.hpp"
#include "imgui_impl_ios.h"

#import "UIKit/UIKit.h"

using namespace My;

void* UIKitApplication::GetMainWindowHandler() { return (void*)NULL; }

void UIKitApplication::CreateMainWindow() {
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    [[maybe_unused]] ImGuiIO& io = ImGui::GetIO();

    ImGui_ImplIOS_Init();

    ImGui::StyleColorsDark();
}

void UIKitApplication::Finalize() {
    ImGui_ImplIOS_Shutdown();

    BaseApplication::Finalize();
}

void UIKitApplication::Tick() {
}
