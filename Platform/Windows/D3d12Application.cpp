#include "D3d12Application.hpp"
#include "D3d/D3d12GraphicsManager.hpp"
#include "MemoryManager.hpp"
#include "AssetLoader.hpp"
#include "SceneManager.hpp"
#include <tchar.h>

using namespace My;

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 32, 0, 0, 960, 540, _T("Game Engine From Scratch (Windows D3d12)"));
	IApplication* g_pApp                = static_cast<IApplication*>(new D3d12Application(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new D3d12GraphicsManager);
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AssetLoader);
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager);
}

void D3d12Application::Tick()
{
    WindowsApplication::Tick();
    g_pGraphicsManager->Clear();
    g_pGraphicsManager->Draw();
    
    // Present the back buffer to the screen since rendering is complete.
    HDC hdc = GetDC(m_hWnd);
    SwapBuffers(hdc);
	ReleaseDC(m_hWnd, hdc);
}

