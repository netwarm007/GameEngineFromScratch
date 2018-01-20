#include <iostream>
#include "InputManager.hpp"
#include "GraphicsManager.hpp"
#include "SceneManager.hpp"
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
}

void InputManager::UpArrowKeyUp()
{
#ifdef DEBUG
    cerr << "[InputManager] Up Arrow Key Up!" << endl;
#endif
}

void InputManager::DownArrowKeyDown()
{
#ifdef DEBUG
    cerr << "[InputManager] Down Arrow Key Down!" << endl;
#endif
}

void InputManager::DownArrowKeyUp()
{
#ifdef DEBUG
    cerr << "[InputManager] Down Arrow Key Up!" << endl;
#endif
}

void InputManager::LeftArrowKeyDown()
{
#ifdef DEBUG
    cerr << "[InputManager] Left Arrow Key Down!" << endl;
#endif
}

void InputManager::LeftArrowKeyUp()
{
#ifdef DEBUG
    cerr << "[InputManager] Left Arrow Key Up!" << endl;
#endif
}

void InputManager::RightArrowKeyDown()
{
#ifdef DEBUG
    cerr << "[InputManager] Right Arrow Key Down!" << endl;
#endif
}

void InputManager::RightArrowKeyUp()
{
#ifdef DEBUG
    cerr << "[InputManager] Right Arrow Key Up!" << endl;
#endif
}

void InputManager::ResetKeyDown()
{
#ifdef DEBUG
    cerr << "[InputManager] Reset Key Down!" << endl;
#endif
    g_pSceneManager->ResetScene();
}


