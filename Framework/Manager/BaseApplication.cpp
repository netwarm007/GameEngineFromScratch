#include "BaseApplication.hpp"

#include <cassert>
#include <iostream>

using namespace My;
using namespace std;

// Parse command line, read configuration, initialize all sub modules
int BaseApplication::Initialize() {
    cout << m_Config;

    return runtime_modules.Initialize();
}

// Finalize all sub modules and clean up all runtime temporary files.
void BaseApplication::Finalize() {
    runtime_modules.Finalize();
}

// One cycle of the main loop
void BaseApplication::Tick() {
    runtime_modules.Tick();
}

void BaseApplication::SetCommandLineParameters(int argc, char** argv) {
    m_nArgC = argc;
    m_ppArgV = argv;
}

int BaseApplication::GetCommandLineArgumentsCount() const { return m_nArgC; }

const char* BaseApplication::GetCommandLineArgument(int index) const {
    assert(index < m_nArgC);
    return m_ppArgV[index];
}

bool BaseApplication::IsQuit() const { return m_bQuit; }

void BaseApplication::RegisterManagerModule(IGraphicsManager* mgr) {
    runtime_modules.Register(*mgr);
    m_pGraphicsManager = mgr;
    mgr->SetAppPointer(this);
}

void BaseApplication::RegisterManagerModule(IMemoryManager* mgr) {
    runtime_modules.Register(*mgr);
    m_pMemoryManager = mgr;
    mgr->SetAppPointer(this);
}

void BaseApplication::RegisterManagerModule(IAssetLoader* mgr) {
    runtime_modules.Register(*mgr);
    m_pAssetLoader = mgr;
    mgr->SetAppPointer(this);
}

void BaseApplication::RegisterManagerModule(IInputManager* mgr) {
    runtime_modules.Register(*mgr);
    m_pInputManager = mgr;
    mgr->SetAppPointer(this);
}

void BaseApplication::RegisterManagerModule(ISceneManager* mgr) {
    runtime_modules.Register(*mgr);
    m_pSceneManager = mgr;
    mgr->SetAppPointer(this);
}

#ifdef DEBUG
void BaseApplication::RegisterManagerModule(IDebugManager* mgr) {
    runtime_modules.Register(*mgr);
    m_pDebugManager = mgr;
    mgr->SetAppPointer(this);
}
#endif

void BaseApplication::RegisterManagerModule(IAnimationManager* mgr) {
    runtime_modules.Register(*mgr);
    m_pAnimationManager = mgr;
    mgr->SetAppPointer(this);
}

void BaseApplication::RegisterManagerModule(IPhysicsManager* mgr) {
    runtime_modules.Register(*mgr);
    m_pPhysicsManager = mgr;
    mgr->SetAppPointer(this);
}

void BaseApplication::RegisterManagerModule(IPipelineStateManager* mgr) {
    runtime_modules.Register(*mgr);
    m_pPipelineStateManager = mgr;
    mgr->SetAppPointer(this);
}

void BaseApplication::RegisterManagerModule(IGameLogic* logic) {
    runtime_modules.Register(*logic);
    m_pGameLogic = logic;
    logic->SetAppPointer(this);
}
