#include "RuntimeRoot.hpp"

#include <iostream>
#include <utility>

#include "EngineAllocation.hpp"
#include "FrameScheduler.hpp"

using namespace My;

RuntimeRoot::RuntimeRoot() = default;
RuntimeRoot::~RuntimeRoot() { ReleaseMemorySystem(); }

void RuntimeRoot::SetHost(IApplication& application, IPlatformHost& platform,
                          IRenderBackend& backend) {
    m_Application = &application;
    m_Platform = &platform;
    m_Backend = &backend;
}

void RuntimeRoot::RegisterPersistentModule(IRuntimeModule& module) {
    module.SetAppPointer(m_Application);
    m_Persistent.Register(module);
}

void RuntimeRoot::RegisterSessionModule(IRuntimeModule& module) {
    module.SetAppPointer(m_Application);
    m_Session.Register(module);
}

void RuntimeRoot::SetFramePlan(std::vector<IRuntimeModule*> plan) {
    m_FramePlan = std::move(plan);
}

int RuntimeRoot::Initialize() {
    if (!m_Application || !m_Platform || !m_Backend) {
        std::cerr << "Runtime root requires a host before initialization\n";
        return -1;
    }

    const int graphics = m_Backend->InitializeGraphicsResources();
    if (graphics != 0) {
        std::cerr << "Graphics resource initialization failed (error "
                  << graphics << ")\n";
        return graphics;
    }
    m_GraphicsInitialized = true;

    const int gui = m_Platform->InitializeGuiPlatform();
    if (gui != 0) {
        std::cerr << "GUI platform initialization failed (error " << gui
                  << ")\n";
        return gui;
    }
    m_GuiPlatformInitialized = true;

#ifdef USE_ENGINE_ALLOCATORS
    MemorySystem::Config config;
#if defined(ENABLE_ALLOCATOR_TRACKING) || !defined(NDEBUG)
    config.tracking = true;  // always on in Debug builds
#endif
    InitializeMemorySystem(config);
#endif

    const int persistent =
        m_Persistent.Initialize([this] { m_Backend->WaitForGraphicsIdle(); });
    if (persistent != 0) return persistent;

    const int session =
        m_Session.Activate([this] { m_Backend->WaitForGraphicsIdle(); });
    if (session != 0) {
        m_Persistent.Finalize();
        return session;
    }
    return 0;
}

void RuntimeRoot::Finalize() {
    if (m_Backend) m_Backend->WaitForGraphicsIdle();
    m_Session.Retire();
    m_Persistent.Finalize();
    if (m_GuiPlatformInitialized && m_Platform) {
        m_Platform->FinalizeGuiPlatform();
        m_GuiPlatformInitialized = false;
    }
    if (m_GraphicsInitialized && m_Backend) {
        m_Backend->FinalizeGraphicsResources();
        m_GraphicsInitialized = false;
    }
    if (m_Platform) {
        m_Platform->FinalizePlatformResources();
        m_Platform->ShutdownPlatform();
    }
    ReleaseMemorySystem();
}

void RuntimeRoot::Tick() {
    if (m_Platform) m_Platform->ProcessEvents();
    BeginFrame();
    TickFrame(m_FramePlan);
    if (m_Backend) m_Backend->PresentFrame();
}

void RuntimeRoot::RetireSessionForReconstruction(bool recreateWindow) {
    if (m_Backend) m_Backend->WaitForGraphicsIdle();
    m_Session.Retire();
    // GUI platform callbacks are bound to the native window: unbind them only
    // when the window is actually replaced. Keeping them bound across a
    // same-window reconstruction also preserves the platform backend's frame
    // timing (a fresh bind would yield a zero first DeltaTime).
    if (recreateWindow && m_GuiPlatformInitialized && m_Platform) {
        m_Platform->FinalizeGuiPlatform();
        m_GuiPlatformInitialized = false;
    }
    if (m_GraphicsInitialized && m_Backend) {
        m_Backend->FinalizeGraphicsResources();
        m_GraphicsInitialized = false;
    }
    if (recreateWindow && m_Platform) {
        m_Platform->FinalizePlatformResources();
    }
}

int RuntimeRoot::PrepareReconstructedSession(bool windowRecreated) {
    if (!m_Backend || !m_Platform) return -1;
    const int graphics = m_Backend->InitializeGraphicsResources();
    if (graphics != 0) {
        std::cerr << "Reconstructed graphics initialization failed (error "
                  << graphics << ")\n";
        return graphics;
    }
    m_GraphicsInitialized = true;
    if (windowRecreated) {
        const int gui = m_Platform->InitializeGuiPlatform();
        if (gui != 0) {
            std::cerr << "Reconstructed GUI platform initialization failed (error "
                      << gui << ")\n";
            m_Backend->FinalizeGraphicsResources();
            m_GraphicsInitialized = false;
            return gui;
        }
        m_GuiPlatformInitialized = true;
    }
    return 0;
}

bool RuntimeRoot::InitializeMemorySystem(const MemorySystem::Config& config) {
    if (m_pMemorySystem) return m_pMemorySystem->IsConfigured();
    auto system = std::make_unique<MemorySystem>(config);
    if (!system->IsConfigured()) {
        std::cerr << "Engine allocators unavailable; using default heap\n";
        return false;
    }
    m_pMemorySystem = std::move(system);
    SetEngineAllocator(&m_pMemorySystem->Allocator());
    return true;
}

void RuntimeRoot::ReleaseMemorySystem() {
    if (!m_pMemorySystem) return;
    SetEngineAllocator(nullptr);
    m_pMemorySystem.reset();
}