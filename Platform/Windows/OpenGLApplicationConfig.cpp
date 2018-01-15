#include <tchar.h>
#include "OpenGLApplication.hpp"
#include "OpenGL/OpenGLGraphicsManager.hpp"

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 0, 960, 540, _T("Game Engine From Scratch (Win32 + OpenGL)"));
	IApplication* g_pApp                = static_cast<IApplication*>(new OpenGLApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new OpenGLGraphicsManager);
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AssetLoader);
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager);
    InputManager*    g_pInputManager    = static_cast<InputManager*>(new InputManager);
}

