#include <iostream>

#include "D3d12Application.hpp"
#include "D3d/D3d12GraphicsManager.hpp"
#include "imgui.h"

using namespace My;

int main() {
    ImGui::CreateContext();
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 1, 320, 240,
                            "DirectX 12 GUI Texture Test");
    D3d12Application app(config);
    D3d12GraphicsManager graphics;
    app.RegisterManagerModule(&graphics);
    app.CreateMainWindow();
    if (app.Initialize() != 0) {
        std::cerr << "Failed to initialize graphics manager\n";
        return 1;
    }

    Image image;
    image.Width = image.Height = 1;
    image.bitcount = 32;
    image.bitdepth = 8;
    image.pitch = image.data_size = 4;
    image.pixel_format = PIXEL_FORMAT::RGBA8;
    image.data = new uint8_t[4]{255, 0, 0, 255};

    auto first = graphics.CreateTexture(image);
    auto second = graphics.CreateTexture(image);
    const auto firstHandle = first.handler;
    bool ok = first.buff && first.handler && second.buff && second.handler &&
              first.handler != second.handler;
    graphics.BindDebugTexture(first, 1);
    graphics.ReleaseTexture(first);
    auto replacement = graphics.CreateTexture(image);
    ok = ok && replacement.handler == firstHandle;
    graphics.ReleaseTexture(second);
    graphics.ReleaseTexture(replacement);
    if (!ok) std::cerr << "GUI texture descriptors were not allocated or reused correctly\n";

    app.Finalize();
    ImGui::DestroyContext();
    return ok ? 0 : 1;
}
