#include "InputManager.hpp"

#include <iostream>

#include "DebugManager.hpp"
#include "GraphicsManager.hpp"
#include "IGameLogic.hpp"
#include "SceneManager.hpp"
#include "geommath.hpp"

using namespace My;
using namespace std;

int InputManager::Initialize() { return 0; }

void InputManager::Finalize() {}

void InputManager::Tick() {}

void InputManager::UpArrowKeyDown() {
    g_pGameLogic->OnUpKeyDown();
    if (!m_bUpKeyPressed) {
        g_pGameLogic->OnUpKey();
        m_bUpKeyPressed = true;
    }
}

void InputManager::UpArrowKeyUp() {
    g_pGameLogic->OnUpKeyUp();
    m_bUpKeyPressed = false;
}

void InputManager::DownArrowKeyDown() {
    g_pGameLogic->OnDownKeyDown();
    if (!m_bDownKeyPressed) {
        g_pGameLogic->OnDownKey();
        m_bDownKeyPressed = true;
    }
}

void InputManager::DownArrowKeyUp() {
    g_pGameLogic->OnDownKeyUp();
    m_bDownKeyPressed = false;
}

void InputManager::LeftArrowKeyDown() {
    g_pGameLogic->OnLeftKeyDown();
    if (!m_bLeftKeyPressed) {
        g_pGameLogic->OnLeftKey();
        m_bLeftKeyPressed = true;
    }
}

void InputManager::LeftArrowKeyUp() {
    g_pGameLogic->OnLeftKeyUp();
    m_bLeftKeyPressed = false;
}

void InputManager::RightArrowKeyDown() {
    g_pGameLogic->OnRightKeyDown();
    if (!m_bRightKeyPressed) {
        g_pGameLogic->OnRightKey();
        m_bRightKeyPressed = true;
    }
}

void InputManager::RightArrowKeyUp() {
    g_pGameLogic->OnRightKeyUp();
    m_bRightKeyPressed = false;
}

void InputManager::AsciiKeyDown(char keycode) {
    switch (keycode) {
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

void InputManager::AsciiKeyUp(char keycode) {
    switch (keycode) {
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

void InputManager::LeftMouseButtonDown() {}

void InputManager::LeftMouseButtonUp() {}

void InputManager::LeftMouseDrag(int deltaX, int deltaY) {
    g_pGameLogic->OnAnalogStick(0, (float)deltaX, (float)deltaY);
}

void InputManager::RightMouseButtonDown() {}

void InputManager::RightMouseButtonUp() {}

void InputManager::RightMouseDrag(int deltaX, int deltaY) {
    g_pGameLogic->OnAnalogStick(1, (float)deltaX, (float)deltaY);
}
