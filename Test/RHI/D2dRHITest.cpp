#include "config.h"

#include "AssetLoader.hpp"
#include "D2dApplication.hpp"
#include "PVR.hpp"

using namespace My;

template <class T>
inline void SafeRelease(T **ppInterfaceToRelease)
{
    if (*ppInterfaceToRelease != nullptr)
    {
        (*ppInterfaceToRelease)->Release();

        (*ppInterfaceToRelease) = nullptr;
    }
};

int main() {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                            "Direct2D RHI Test");
    D2dApplication app(config);

    AssetLoader asset_loader;

    assert(asset_loader.Initialize() == 0 && "Asset Loader Initialize Failed!");

    // 创建窗口
    {
        assert(app.Initialize() == 0 && "App Initialize Failed!");

        app.CreateMainWindow();
    }

    auto& rhi = app.GetRHI();

    // 创建图形管道
    rhi.CreateGraphicsResources();

    // 创建画布颜色
    Vector4f clearColor {0.3f, 0.3f, 0.3f, 1.0f};

    // 创建画笔
    auto *pLightSlateGrayBrush = rhi.CreateSolidColorBrush({0.3f, 0.3f, 0.3f});
    auto *pCornflowerBlueBrush = rhi.CreateSolidColorBrush({0.0f, 0.0f, 0.5f});

    D2dRHI::DestroyResourceFunc destroyResourceFunc =
        [&pLightSlateGrayBrush, &pCornflowerBlueBrush]() {
            SafeRelease(&pLightSlateGrayBrush);
            SafeRelease(&pCornflowerBlueBrush);
        };

    rhi.DestroyResourceCB(destroyResourceFunc);

    // 主消息循环
    while (!app.IsQuit()) {
        app.Tick();

        // 绘制一帧
        rhi.BeginFrame();

        rhi.ClearCanvas(clearColor);

        rhi.EndFrame();
    }

    app.Finalize();

    return 0;
}