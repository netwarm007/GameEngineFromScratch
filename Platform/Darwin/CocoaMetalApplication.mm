#import "MetalView.h"

#include "CocoaMetalApplication.h"
#include "imgui_impl_osx.h"

using namespace My;

void CocoaMetalApplication::CreateMainWindow() {
    CocoaApplication::CreateMainWindow();

    @autoreleasepool {
        MetalView* pView = [[MetalView new]
            initWithFrame:CGRectMake(0, 0, m_Config.screenWidth, m_Config.screenHeight) pApp:this];

        [m_pWindow setContentView:pView];

        ImGui_ImplOSX_Init(pView);
    }
}

void CocoaMetalApplication::Tick() {
    CocoaApplication::Tick();

}

void CocoaMetalApplication::Finalize() {
    ImGui_ImplOSX_Shutdown();

    CocoaApplication::Finalize();
}
