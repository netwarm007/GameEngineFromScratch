#include <iostream>
#include "InputManager.hpp"
#include "GraphicsManager.hpp"
#include "SceneManager.hpp"
#include "IGameLogic.hpp"
#include "DebugManager.hpp"
#include "geommath.hpp"

using namespace My;
using namespace std;

int InputManager::Initialize()
{
    return 0;
}

void InputManager::Finalize()
{
}

void InputManager::Tick()
{
}

void InputManager::UpArrowKeyDown()
{
#ifdef DEBUG
    cerr << "[InputManager] Up Arrow Key Down!" << endl;
#endif
    g_pGameLogic->OnUpKeyDown();
    if (!m_bUpKeyPressed)
    {
        g_pGameLogic->OnUpKey();
        m_bUpKeyPressed = true;
    }
}

void InputManager::UpArrowKeyUp()
{
#ifdef DEBUG
    cerr << "[InputManager] Up Arrow Key Up!" << endl;
#endif
    g_pGameLogic->OnUpKeyUp();
    m_bUpKeyPressed = false;
}

void InputManager::DownArrowKeyDown()
{
#ifdef DEBUG
    cerr << "[InputManager] Down Arrow Key Down!" << endl;
#endif
    g_pGameLogic->OnDownKeyDown();
    if (!m_bDownKeyPressed)
    {
        g_pGameLogic->OnDownKey();
        m_bDownKeyPressed = true;
    }
}

void InputManager::DownArrowKeyUp()
{
#ifdef DEBUG
    cerr << "[InputManager] Down Arrow Key Up!" << endl;
#endif
    g_pGameLogic->OnDownKeyUp();
    m_bDownKeyPressed = false;
}

void InputManager::LeftArrowKeyDown()
{
#ifdef DEBUG
    cerr << "[InputManager] Left Arrow Key Down!" << endl;
#endif
    g_pGameLogic->OnLeftKeyDown();
    if (!m_bLeftKeyPressed)
    {
        g_pGameLogic->OnLeftKey();
        m_bLeftKeyPressed = true;
    }
}

void InputManager::LeftArrowKeyUp()
{
#ifdef DEBUG
    cerr << "[InputManager] Left Arrow Key Up!" << endl;
#endif
    g_pGameLogic->OnLeftKeyUp();
    m_bLeftKeyPressed = false;
}

void InputManager::RightArrowKeyDown()
{
#ifdef DEBUG
    cerr << "[InputManager] Right Arrow Key Down!" << endl;
#endif
    g_pGameLogic->OnRightKeyDown();
    if(!m_bRightKeyPressed)
    {
        g_pGameLogic->OnRightKey();
        m_bRightKeyPressed = true;
    }
}

void InputManager::RightArrowKeyUp()
{
#ifdef DEBUG
    cerr << "[InputManager] Right Arrow Key Up!" << endl;
#endif
    g_pGameLogic->OnRightKeyUp();
    m_bRightKeyPressed = false;
}

void InputManager::ResetKeyDown()
{
#ifdef DEBUG
    cerr << "[InputManager] Reset Key Down!" << endl;
#endif
    g_pSceneManager->ResetScene();
}

void InputManager::ResetKeyUp()
{
#ifdef DEBUG
    cerr << "[InputManager] Reset Key Up!" << endl;
#endif
}

#ifdef DEBUG
void InputManager::DebugKeyDown()
{
    cerr << "[InputManager] Debug Key Down!" << endl;
    g_pDebugManager->ToggleDebugInfo();
}
#endif

#ifdef DEBUG
void InputManager::DebugKeyUp()
{
    cerr << "[InputManager] Debug Key Up!" << endl;
}
#endif

