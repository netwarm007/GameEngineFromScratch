#include "Platform/CrossPlatformApp.hpp"

using namespace My;

int main() {
    int result;

    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600, "Basic Window Test");

    auto pApp = My::CreateApplication(config);

    result = pApp->Initialize();

    pApp->CreateMainWindow();

    while(!pApp->IsQuit()) {
        pApp->Tick();
    }

    pApp->Finalize();

    delete pApp;

    return result;
}