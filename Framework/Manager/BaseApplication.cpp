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

void BaseApplication::RegisterManagerModule(IGraphicsManager* mgr,
                                            IRuntimeModule* module) {
    m_pGraphicsManager = mgr;
    module->SetAppPointer(this);
    runtime_modules.push_back(module);
}

void BaseApplication::RegisterManagerModule(IMemoryManager* mgr,
                                            IRuntimeModule* module) {
    m_pMemoryManager = mgr;
    module->SetAppPointer(this);
    runtime_modules.push_back(module);
}

void BaseApplication::RegisterManagerModule(IAssetLoader* mgr,
                                            IRuntimeModule* module) {
    m_pAssetLoader = mgr;
    module->SetAppPointer(this);
    runtime_modules.push_back(module);
}

void BaseApplication::RegisterManagerModule(IInputManager* mgr,
                                            IRuntimeModule* module) {
    m_pInputManager = mgr;
    module->SetAppPointer(this);
    runtime_modules.push_back(module);
}

void BaseApplication::RegisterManagerModule(ISceneManager* mgr,
                                            IRuntimeModule* module) {
    m_pSceneManager = mgr;
    module->SetAppPointer(this);
    runtime_modules.push_back(module);
}

#ifdef DEBUG
void BaseApplication::RegisterManagerModule(IDebugManager* mgr,
                                            IRuntimeModule* module) {
    m_pDebugManager = mgr;
    module->SetAppPointer(this);
    runtime_modules.push_back(module);
}
#endif

void BaseApplication::RegisterManagerModule(IAnimationManager* mgr,
                                            IRuntimeModule* module) {
    m_pAnimationManager = mgr;
    module->SetAppPointer(this);
    runtime_modules.push_back(module);
}

void BaseApplication::RegisterManagerModule(IPhysicsManager* mgr,
                                            IRuntimeModule* module) {
    m_pPhysicsManager = mgr;
    module->SetAppPointer(this);
    runtime_modules.push_back(module);
}

void BaseApplication::RegisterManagerModule(IPipelineStateManager* mgr,
                                            IRuntimeModule* module) {
    m_pPipelineStateManager = mgr;
    module->SetAppPointer(this);
    runtime_modules.push_back(module);
}

void BaseApplication::RegisterManagerModule(IGameLogic* logic,
                                            IRuntimeModule* module) {
    m_pGameLogic = logic;
    module->SetAppPointer(this);
    runtime_modules.push_back(module);
}
