#include <iostream>

#include "D3d12Application.hpp"
#include "D3d/D3d12GraphicsManager.hpp"
#include "imgui.h"

using namespace My;

namespace {
class LaterModule : public IRuntimeModule {
   public:
    int Initialize() override { return result; }
    void Finalize() override {}
    void Tick() override {}
    int result = 17;
};
}  // namespace

int main() {
    ImGui::CreateContext();
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 1, 320, 240,
                            "DirectX 12 GUI Texture Test");
    D3d12Application app(config);
    D3d12GraphicsManager graphics;
    LaterModule later;
    app.RegisterManagerModule(&graphics);
    app.RegisterManagerModule(&later);
    app.CreateMainWindow();
    if (!app.GetMainWindowHandler() || app.GetRHI().GetDevice() ||
        ImGui::GetIO().BackendPlatformUserData) {
        std::cerr << "Window creation must not initialize D3D12 or ImGui backends\n";
        app.Finalize();
        ImGui::DestroyContext();
        return 1;
    }
    int borrowedBackend = 0;
    ImGui::GetIO().BackendRendererUserData = &borrowedBackend;
    const int rejected = app.Initialize();
    const bool preserved =
        ImGui::GetIO().BackendRendererUserData == &borrowedBackend;
    ImGui::GetIO().BackendRendererUserData = nullptr;
    if (rejected == 0 || !preserved) {
        std::cerr << "A borrowed renderer backend must be rejected without mutation\n";
        app.Finalize();
        ImGui::DestroyContext();
        return 1;
    }
    if (app.Initialize() != 17 ||
        ImGui::GetIO().BackendRendererUserData ||
        !ImGui::GetIO().BackendPlatformUserData) {
        std::cerr << "Module rollback must release the renderer but retain platform input\n";
        app.Finalize();
        ImGui::DestroyContext();
        return 1;
    }
    later.result = 0;
    if (app.Initialize() != 0) {
        std::cerr << "Failed to initialize graphics manager\n";
        app.Finalize();
        ImGui::DestroyContext();
        return 1;
    }

    auto* rendererBackend = ImGui::GetIO().BackendRendererUserData;
    bool ok = rendererBackend && ImGui::GetIO().BackendPlatformUserData &&
              graphics.Initialize() != 0 &&
              ImGui::GetIO().BackendRendererUserData == rendererBackend;
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
    ok = ok && first.buff && first.handler && second.buff && second.handler &&
         first.handler != second.handler;
    graphics.BindDebugTexture(first, 1);
    graphics.ReleaseTexture(first);
    auto replacement = graphics.CreateTexture(image);
    ok = ok && replacement.handler == firstHandle;
    graphics.ReleaseTexture(second);
    graphics.ReleaseTexture(replacement);
    if (!ok) std::cerr << "GUI texture descriptors were not allocated or reused correctly\n";

    app.Finalize();
    graphics.Finalize();
    app.Finalize();
    ok = ok && !ImGui::GetIO().BackendRendererUserData &&
         !ImGui::GetIO().BackendPlatformUserData;
    ImGui::DestroyContext();
    return ok ? 0 : 1;
}
