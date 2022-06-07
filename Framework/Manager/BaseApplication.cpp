#include "BaseApplication.hpp"

#include <cassert>
#include <iostream>

using namespace My;
using namespace std;

// Parse command line, read configuration, initialize all sub modules
int BaseApplication::Initialize() {
    int ret = 0;

    cout << m_Config;

    for (auto module : runtime_modules) {
        if ((ret = module->Initialize()) != 0) {
            std::cerr << "Module initialize failed!\n";
            break;
        }
    }

    return ret;
}

// Finalize all sub modules and clean up all runtime temporary files.
void BaseApplication::Finalize() {
    for (auto module : runtime_modules) {
        module->Finalize();
    }

    runtime_modules.clear();
}

// One cycle of the main loop
void BaseApplication::Tick() {
    for (auto module : runtime_modules) {
        module->Tick();
    }
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
    m_pGraphicsManager = mgr;
    mgr->SetAppPointer(this);
    runtime_modules.push_back(mgr);
}

void BaseApplication::RegisterManagerModule(IMemoryManager* mgr) {
    m_pMemoryManager = mgr;
    mgr->SetAppPointer(this);
    runtime_modules.push_back(mgr);
}

void BaseApplication::RegisterManagerModule(IAssetLoader* mgr) {
    m_pAssetLoader = mgr;
    mgr->SetAppPointer(this);
    runtime_modules.push_back(mgr);
}

void BaseApplication::RegisterManagerModule(IInputManager* mgr) {
    m_pInputManager = mgr;
    mgr->SetAppPointer(this);
    runtime_modules.push_back(mgr);
}

void BaseApplication::RegisterManagerModule(ISceneManager* mgr) {
    m_pSceneManager = mgr;
    mgr->SetAppPointer(this);
    runtime_modules.push_back(mgr);
}

#ifdef DEBUG
void BaseApplication::RegisterManagerModule(IDebugManager* mgr) {
    m_pDebugManager = mgr;
    mgr->SetAppPointer(this);
    runtime_modules.push_back(mgr);
}
#endif

void BaseApplication::RegisterManagerModule(IAnimationManager* mgr) {
    m_pAnimationManager = mgr;
    mgr->SetAppPointer(this);
    runtime_modules.push_back(mgr);
}

void BaseApplication::RegisterManagerModule(IPhysicsManager* mgr) {
    m_pPhysicsManager = mgr;
    mgr->SetAppPointer(this);
    runtime_modules.push_back(mgr);
}

void BaseApplication::RegisterManagerModule(IPipelineStateManager* mgr) {
    m_pPipelineStateManager = mgr;
    mgr->SetAppPointer(this);
    runtime_modules.push_back(mgr);
}

void BaseApplication::RegisterManagerModule(IGameLogic* logic) {
    m_pGameLogic = logic;
    logic->SetAppPointer(this);
    runtime_modules.push_back(logic);
}
