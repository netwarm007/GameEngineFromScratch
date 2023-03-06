#include "config.h"

#include "AssetLoader.hpp"
#include "D2dApplication.hpp"
#include "PVR.hpp"
#include "AssetLoader.hpp"

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

    // 定义资源
    ID2D1SolidColorBrush *pLightSlateGrayBrush;
    ID2D1SolidColorBrush *pCornflowerBlueBrush;
    ID2D1Bitmap          *pBitmap;

    D2dRHI::CreateResourceFunc createResourceFunc =
        [&rhi, &pLightSlateGrayBrush, &pCornflowerBlueBrush, &pBitmap]() {
            pLightSlateGrayBrush = rhi.CreateSolidColorBrush({0.3f, 0.3f, 0.3f});
            pCornflowerBlueBrush = rhi.CreateSolidColorBrush({0.0f, 0.0f, 0.5f});

            // 读取贴图
            {
                int error = 0;

                AssetLoader assetLoader;

                error = assetLoader.Initialize();

                if (!error) {
                    Buffer buf;
                    buf = assetLoader.SyncOpenAndReadBinary(
                        "Textures/viking_room.pvr");

                    PVR::PvrParser astc_parser;

                    auto img = astc_parser.Parse(buf);

                    pBitmap = rhi.CreateBitmap(img);
                }

                assetLoader.Finalize();
            }
        };

    // 登记资源创建回调函数
    rhi.CreateResourceCB(createResourceFunc);

    D2dRHI::DestroyResourceFunc destroyResourceFunc =
        [&pLightSlateGrayBrush, &pCornflowerBlueBrush, &pBitmap]() {
            SafeRelease(&pLightSlateGrayBrush);
            SafeRelease(&pCornflowerBlueBrush);
            SafeRelease(&pBitmap);
        };

    // 登记资源销毁回调函数
    rhi.DestroyResourceCB(destroyResourceFunc);

    // 创建图形资源
    rhi.CreateGraphicsResources();

    // 创建画布颜色
    Vector4f clearColor {1.0f, 1.0f, 1.0f, 1.0f};

    // 主消息循环
    while (!app.IsQuit()) {
        app.Tick();

        // 绘制一帧
        rhi.BeginFrame();

        rhi.ClearCanvas(clearColor);

        auto canvasSize = rhi.GetCanvasSize();

        for (int x = 0; x < canvasSize[0]; x += 10) {
            rhi.DrawLine({static_cast<float>(x), 0.0f}, {static_cast<float>(x), canvasSize[1]}, pLightSlateGrayBrush, 0.5f);
        }

        for (int y = 0; y < canvasSize[1]; y += 10) {
            rhi.DrawLine({0.0f, static_cast<float>(y)}, {canvasSize[0], static_cast<float>(y)}, pLightSlateGrayBrush, 0.5f);
        }

        auto bmpSize = pBitmap->GetSize();
        rhi.DrawBitmap({100.0f, 100.0f}, {100.0f + bmpSize.width, 100.0f + bmpSize.height}, pBitmap);

        rhi.EndFrame();
    }

    app.Finalize();

    return 0;
}