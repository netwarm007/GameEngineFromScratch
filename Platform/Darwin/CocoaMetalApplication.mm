#import "MetalView.h"

#include "CocoaMetalApplication.h"
#include "imgui_impl_osx.h"

using namespace My;

void CocoaMetalApplication::CreateMainWindow() {
    CocoaApplication::CreateMainWindow();

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui::StyleColorsDark();

    ImGuiStyle& im_style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        im_style.WindowRounding = 0.0f;
        im_style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    @autoreleasepool {
        MetalView* pView = [[MetalView new]
            initWithFrame:CGRectMake(0, 0, m_Config.screenWidth, m_Config.screenHeight) pApp:this];

        [m_pWindow setContentView:pView];

        ImGui_ImplOSX_Init(pView);
    }
}

void CocoaMetalApplication::Tick() {
    CocoaApplication::Tick();
    //[[m_pWindow contentView] setNeedsDisplay:YES];
}

void CocoaMetalApplication::Finalize() {
    ImGui_ImplOSX_Shutdown();
    ImGui::DestroyContext();

    CocoaApplication::Finalize();
}
