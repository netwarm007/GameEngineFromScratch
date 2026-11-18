#include "InputManager.hpp"

#include <iostream>

#include "IGameLogic.hpp"

using namespace My;
using namespace std;

int InputManager::Initialize() {
    m_bUpKeyPressed = false;
    m_bDownKeyPressed = false;
    m_bLeftKeyPressed = false;
    m_bRightKeyPressed = false;
    return 0;
}

void InputManager::Finalize() {
    m_bUpKeyPressed = false;
    m_bDownKeyPressed = false;
    m_bLeftKeyPressed = false;
    m_bRightKeyPressed = false;
}

void InputManager::Tick() {}

void InputManager::UpArrowKeyDown() {
    m_GameLogic.OnUpKeyDown();
    if (!m_bUpKeyPressed) {
        m_GameLogic.OnUpKey();
        m_bUpKeyPressed = true;
    }
}

void InputManager::UpArrowKeyUp() {
    m_GameLogic.OnUpKeyUp();
    m_bUpKeyPressed = false;
}

void InputManager::DownArrowKeyDown() {
    m_GameLogic.OnDownKeyDown();
    if (!m_bDownKeyPressed) {
        m_GameLogic.OnDownKey();
        m_bDownKeyPressed = true;
    }
}

void InputManager::DownArrowKeyUp() {
    m_GameLogic.OnDownKeyUp();
    m_bDownKeyPressed = false;
}

void InputManager::LeftArrowKeyDown() {
    m_GameLogic.OnLeftKeyDown();
    if (!m_bLeftKeyPressed) {
        m_GameLogic.OnLeftKey();
        m_bLeftKeyPressed = true;
    }
}

void InputManager::LeftArrowKeyUp() {
    m_GameLogic.OnLeftKeyUp();
    m_bLeftKeyPressed = false;
}

void InputManager::RightArrowKeyDown() {
    m_GameLogic.OnRightKeyDown();
    if (!m_bRightKeyPressed) {
        m_GameLogic.OnRightKey();
        m_bRightKeyPressed = true;
    }
}

void InputManager::RightArrowKeyUp() {
    m_GameLogic.OnRightKeyUp();
    m_bRightKeyPressed = false;
}

void InputManager::AsciiKeyDown(char keycode) {
    switch (keycode) {
        case 'd':
            break;
        case 'r':
            break;
        case 'u':
            m_GameLogic.OnButton1Down();
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
            m_GameLogic.OnButton1Up();
            break;
        default:
            cerr << "[InputManager] unhandled key." << endl;
    }
}

void InputManager::LeftMouseButtonDown() {}

void InputManager::LeftMouseButtonUp() {}

void InputManager::LeftMouseDrag(int deltaX, int deltaY) {
    m_GameLogic.OnAnalogStick(0, static_cast<float>(deltaX),
                             static_cast<float>(deltaY));
}

void InputManager::RightMouseButtonDown() {}

void InputManager::RightMouseButtonUp() {}

void InputManager::RightMouseDrag(int deltaX, int deltaY) {
    m_GameLogic.OnAnalogStick(1, static_cast<float>(deltaX),
                             static_cast<float>(deltaY));
}
