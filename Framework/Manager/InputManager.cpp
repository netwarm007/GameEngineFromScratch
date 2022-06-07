#include "InputManager.hpp"

#include <iostream>

#include "BaseApplication.hpp"

using namespace My;
using namespace std;

int InputManager::Initialize() { return 0; }

void InputManager::Finalize() {}

void InputManager::Tick() {}

void InputManager::UpArrowKeyDown() {
    auto pGameLogic = dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();
    if (pGameLogic) {
        pGameLogic->OnUpKeyDown();
        if (!m_bUpKeyPressed) {
            pGameLogic->OnUpKey();
            m_bUpKeyPressed = true;
        }
    }
}

void InputManager::UpArrowKeyUp() {
    auto pGameLogic = dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();
    if (pGameLogic) {
        pGameLogic->OnUpKeyUp();
        m_bUpKeyPressed = false;
    }
}

void InputManager::DownArrowKeyDown() {
    auto pGameLogic = dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();
    if (pGameLogic) {
        pGameLogic->OnDownKeyDown();
        if (!m_bDownKeyPressed) {
            pGameLogic->OnDownKey();
            m_bDownKeyPressed = true;
        }
    }
}

void InputManager::DownArrowKeyUp() {
    auto pGameLogic = dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();
    if (pGameLogic) {
        pGameLogic->OnDownKeyUp();
        m_bDownKeyPressed = false;
    }
}

void InputManager::LeftArrowKeyDown() {
    auto pGameLogic = dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();
    if (pGameLogic) {
        pGameLogic->OnLeftKeyDown();
        if (!m_bLeftKeyPressed) {
            pGameLogic->OnLeftKey();
            m_bLeftKeyPressed = true;
        }
    }
}

void InputManager::LeftArrowKeyUp() {
    auto pGameLogic = dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();
    if (pGameLogic) {
        pGameLogic->OnLeftKeyUp();
        m_bLeftKeyPressed = false;
    }
}

void InputManager::RightArrowKeyDown() {
    auto pGameLogic = dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();
    if (pGameLogic) {
        pGameLogic->OnRightKeyDown();
        if (!m_bRightKeyPressed) {
            pGameLogic->OnRightKey();
            m_bRightKeyPressed = true;
        }
    }
}

void InputManager::RightArrowKeyUp() {
    auto pGameLogic = dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();
    if (pGameLogic) {
        pGameLogic->OnRightKeyUp();
        m_bRightKeyPressed = false;
    }
}

void InputManager::AsciiKeyDown(char keycode) {
    switch (keycode) {
        case 'd':
            break;
        case 'r':
            break;
        case 'u': {
            auto pGameLogic =
                dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();

            if (pGameLogic) {
                pGameLogic->OnButton1Down();
            }
        } break;
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
        case 'u': {
            auto pGameLogic =
                dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();
            if (pGameLogic) {
                pGameLogic->OnButton1Up();
            }
        } break;
        default:
            cerr << "[InputManager] unhandled key." << endl;
    }
}

void InputManager::LeftMouseButtonDown() {}

void InputManager::LeftMouseButtonUp() {}

void InputManager::LeftMouseDrag(int deltaX, int deltaY) {
    auto pGameLogic = dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();
    if (pGameLogic) {
        pGameLogic->OnAnalogStick(0, (float)deltaX, (float)deltaY);
    }
}

void InputManager::RightMouseButtonDown() {}

void InputManager::RightMouseButtonUp() {}

void InputManager::RightMouseDrag(int deltaX, int deltaY) {
    auto pGameLogic = dynamic_cast<BaseApplication*>(m_pApp)->GetGameLogic();
    if (pGameLogic) {
        pGameLogic->OnAnalogStick(1, (float)deltaX, (float)deltaY);
    }
}
