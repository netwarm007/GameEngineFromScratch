#include "BaseApplication.hpp"

#include <cassert>
#include <iostream>

#include "FrameScheduler.hpp"
#include "EngineAllocation.hpp"
#include "MemorySystem.hpp"

using namespace My;
using namespace std;

BaseApplication::BaseApplication() = default;
BaseApplication::BaseApplication(GfxConfiguration& cfg) : m_Config(cfg) {}
BaseApplication::~BaseApplication() { ReleaseMemorySystem(); }

// Parse command line, read configuration, initialize all sub modules
int BaseApplication::Initialize() {
    cout << m_Config;

    if (!m_GraphicsInitialized) {
        const int result = InitializeGraphicsResources();
        if (result != 0) {
            cerr << "Graphics resource initialization failed (error "
                 << result << ")\n";
            return result;
        }
        m_GraphicsInitialized = true;
    }

    if (!m_GuiPlatformInitialized) {
        const int result = InitializeGuiPlatform();
        if (result != 0) {
            cerr << "GUI platform initialization failed (error "
                 << result << ")\n";
            return result;
        }
        m_GuiPlatformInitialized = true;
    }

#ifdef USE_ENGINE_ALLOCATORS
    if (!m_pMemorySystem) {
        MemorySystem::Config config;
#if defined(ENABLE_ALLOCATOR_TRACKING) || !defined(NDEBUG)
        config.tracking = true;  // always on in Debug builds
#endif
        auto system = make_unique<MemorySystem>(config);
        if (system->IsConfigured()) {
            m_pMemorySystem = std::move(system);
            SetEngineAllocator(&m_pMemorySystem->Allocator());
        } else {
            cerr << "Engine allocators unavailable; using default heap\n";
        }
    }
#endif

    const int result =
        runtime_modules.Initialize([this] { WaitForGraphicsIdle(); });
    // The system outlives every module, so pooled objects they hold are
    // released first; only a failed start tears it down here.
    if (result != 0 && runtime_modules.GetInitializedModules().empty()) {
        ReleaseMemorySystem();
    }
    return result;
}

// Finalize all sub modules and clean up all runtime temporary files.
void BaseApplication::Finalize() {
    WaitForGraphicsIdle();
    runtime_modules.Finalize();
    if (m_GuiPlatformInitialized) {
        FinalizeGuiPlatform();
        m_GuiPlatformInitialized = false;
    }
    FinalizeGraphicsResources();
    m_GraphicsInitialized = false;
    FinalizePlatformResources();
    ReleaseMemorySystem();
}

void BaseApplication::ReleaseMemorySystem() {
    if (!m_pMemorySystem) return;
    SetEngineAllocator(nullptr);
    m_pMemorySystem.reset();
}

// One cycle of the main loop
void BaseApplication::Tick() {
    if (m_pMemorySystem) m_pMemorySystem->BeginFrame();
    TickFrame(runtime_modules.GetInitializedModules());
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

void BaseApplication::RegisterManagerModule(IRuntimeModule* module) {
    runtime_modules.Register(*module);
    module->SetAppPointer(this);
}

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
