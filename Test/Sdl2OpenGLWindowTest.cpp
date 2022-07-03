#include "GfxConfiguration.hpp"
#include "config.h"

#include "Platform/Sdl/OpenGLApplication.hpp"

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
    int result;

    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                            "SDL2 OpenGL Application Test");
    OpenGLApplication app(config);
    result |= test(app);

    return result;
}