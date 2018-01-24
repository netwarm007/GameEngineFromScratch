#include <iostream>
#include <tchar.h>
#include "WindowsApplication.hpp"
#include "D2d/D2dGraphicsManager.hpp"
#include "utility.hpp"
#include "BMP.hpp"
#include "JPEG.hpp"
#include "PNG.hpp"

using namespace My;
using namespace std;

namespace My {
    class TestGraphicsManager : public D2dGraphicsManager
    {
        public:
            using D2dGraphicsManager::D2dGraphicsManager;
            void DrawBitmap(const Image image);
        private:
            ID2D1Bitmap* m_pBitmap = nullptr;
    };

    class TestApplication : public WindowsApplication
    {
    public:
        using WindowsApplication::WindowsApplication;

        virtual int Initialize();

        virtual void OnDraw();

    private:
        Image m_Image;
    };
}

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 32, 0, 0, 1024, 512, _T("Texture Load Test (Windows)"));
	IApplication* g_pApp                = static_cast<IApplication*>(new TestApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new TestGraphicsManager);
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AssetLoader);
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager);
    InputManager*    g_pInputManager    = static_cast<InputManager*>(new InputManager);
    PhysicsManager*  g_pPhysicsManager  = static_cast<PhysicsManager*>(new PhysicsManager);
}

int My::TestApplication::Initialize()
{
    int result;

    result = WindowsApplication::Initialize();

    if (result == 0) {
        Buffer buf;

        PngParser  png_parser;
        if (m_nArgC > 1) {
            buf = g_pAssetLoader->SyncOpenAndReadBinary(m_ppArgV[1]);
        } else {
            buf = g_pAssetLoader->SyncOpenAndReadBinary("Textures/eye.png");
        }

        m_Image = png_parser.Parse(buf);
    }

    if (m_Image.bitcount == 24) {
        // DXGI does not have 24bit formats so we have to extend it to 32bit
        uint32_t new_pitch = m_Image.pitch / 3 * 4;
        size_t data_size = new_pitch * m_Image.Height;
        void* data = g_pMemoryManager->Allocate(data_size);
        uint8_t* buf = reinterpret_cast<uint8_t*>(data);
        uint8_t* src = reinterpret_cast<uint8_t*>(m_Image.data);
        for (auto row = 0; row < m_Image.Height; row++) {
            buf = reinterpret_cast<uint8_t*>(data) + row * new_pitch;
            src = reinterpret_cast<uint8_t*>(m_Image.data) + row * m_Image.pitch;
            for (auto col = 0; col < m_Image.Width; col++) {
                *(uint32_t*)buf = *(uint32_t*)src;
                buf[3] = 0;  // set alpha to 0
                buf += 4;
                src += 3;
            }
        }
        g_pMemoryManager->Free(m_Image.data, m_Image.data_size);
        m_Image.data = data;
        m_Image.data_size = data_size;
        m_Image.pitch = new_pitch;
    }

    return result;
}

void My::TestApplication::OnDraw()
{
    dynamic_cast<TestGraphicsManager*>(g_pGraphicsManager)->DrawBitmap(m_Image);
}

void My::TestGraphicsManager::DrawBitmap(const Image image)
{
	HRESULT hr;

    // start build GPU draw command
    m_pRenderTarget->BeginDraw();

    D2D1_BITMAP_PROPERTIES props;
    props.pixelFormat.format = DXGI_FORMAT_R8G8B8A8_UNORM;
    props.pixelFormat.alphaMode = D2D1_ALPHA_MODE_IGNORE;
    props.dpiX = 72.0f;
    props.dpiY = 72.0f;
    SafeRelease(&m_pBitmap);
    hr = m_pRenderTarget->CreateBitmap(D2D1::SizeU(image.Width, image.Height), 
                                                    image.data, image.pitch, props, &m_pBitmap);

    D2D1_SIZE_F rtSize = m_pRenderTarget->GetSize();
    D2D1_SIZE_F bmpSize = m_pBitmap->GetSize();

    D2D1_RECT_F source_rect = D2D1::RectF(
                     0,
                     0,
                     bmpSize.width,
                     bmpSize.height
                     );

    float aspect = bmpSize.width / bmpSize.height;
	float dest_height = rtSize.height;
	float dest_width = rtSize.height * aspect;

    D2D1_RECT_F dest_rect = D2D1::RectF(
                     0,
                     0,
                     dest_width,
                     dest_height 
                     );

    m_pRenderTarget->DrawBitmap(m_pBitmap, dest_rect, 1.0f, D2D1_BITMAP_INTERPOLATION_MODE_NEAREST_NEIGHBOR, source_rect);

    // end GPU draw command building
    m_pRenderTarget->EndDraw();
}


