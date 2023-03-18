#include "Platform/CrossPlatformGfxApp.hpp"

using namespace My;

int test(BaseApplication& app) {
    auto result = app.Initialize();

    if (result == 0) {
        app.CreateMainWindow();
        while (!app.IsQuit()) {
            app.Tick();
        }

        app.Finalize();
    }

    return result;
}

int main() {
    int result = 0;

    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600, "Basic Graphics Window Test");

    auto app = My::CreateApplication(config);
    result |= test(*app);

    delete app;

    return result;
}