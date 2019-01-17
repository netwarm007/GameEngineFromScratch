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

void InputManager::AsciiKeyDown(char keycode)
{
#ifdef DEBUG
    cerr << "[InputManager] ASCII Key Down! (" << keycode << ")" << endl;
#endif
    switch (keycode)
    {
        case 'd':
#ifdef DEBUG
            g_pDebugManager->ToggleDebugInfo();
#endif
            break;
        case 'r':
            g_pSceneManager->ResetScene();
            break;
        case 'u':
            g_pGameLogic->OnButton1Down();
            break;
        default:
            cerr << "[InputManager] unhandled key." << endl;
    }
}

void InputManager::AsciiKeyUp(char keycode)
{
#ifdef DEBUG
    cerr << "[InputManager] ASCII Key Up! (" << keycode << ")" << endl;
#endif
    switch (keycode)
    {
        case 'd':
            break;
        case 'r':
            break;
        case 'u':
            g_pGameLogic->OnButton1Up();
            break;
        default:
            cerr << "[InputManager] unhandled key." << endl;
    }
}

void InputManager::LeftMouseButtonDown()
{
#ifdef DEBUG
    cerr << "[InputManager] Left Mouse Button Down!" << endl;
#endif
}

void InputManager::LeftMouseButtonUp()
{
#ifdef DEBUG
    cerr << "[InputManager] Left Mouse Button Up!" << endl;
#endif
}

void InputManager::LeftMouseDrag(int deltaX, int deltaY)
{
#ifdef DEBUG
    cerr << "[InputManager] Left Mouse Dragged! (" 
        << deltaX << ","
        << deltaY << ")"
        << endl;
#endif
    g_pGameLogic->OnAnalogStick(0, (float)deltaX, (float)deltaY);
}
