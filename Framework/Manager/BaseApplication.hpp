#pragma once
#include "IAnimationManager.hpp"
#include "IApplication.hpp"
#include "IAssetLoader.hpp"
#include "IDebugManager.hpp"
#include "IGameLogic.hpp"
#include "IGraphicsManager.hpp"
#include "IInputManager.hpp"
#include "IMemoryManager.hpp"
#include "IPhysicsManager.hpp"
#include "IPipelineStateManager.hpp"
#include "IRuntimeModule.hpp"
#include "ISceneManager.hpp"
#include "RuntimeModuleGroup.hpp"

#include <memory>

namespace My {
class MemorySystem;

class BaseApplication : _implements_ IApplication {
   public:
    // Defined out of line: MemorySystem is incomplete here.
    BaseApplication();
    explicit BaseApplication(GfxConfiguration& cfg);
    ~BaseApplication() override;
    virtual int Initialize();
    virtual void Finalize();
    // One cycle of the main loop
    virtual void Tick();

    void SetCommandLineParameters(int argc, char** argv) override;
    [[nodiscard]] int GetCommandLineArgumentsCount() const override;
    [[nodiscard]] const char* GetCommandLineArgument(int index) const override;

    [[nodiscard]] bool IsQuit() const override;
    void RequestQuit() override { m_bQuit = true; }

    [[nodiscard]] inline const GfxConfiguration GetConfiguration()
        const override {
        return m_Config;
    }

    // Render conventions follow the active render session (M4 switching):
    // update the host configuration before the restored scene prepares
    // camera/light data.
    void SetRenderConvention(bool fixOpenGLPerspectiveMatrix) {
        m_Config.fixOpenGLPerspectiveMatrix = fixOpenGLPerspectiveMatrix;
    }

    void GetFramebufferSize(uint32_t& width, uint32_t& height) override {}

    void CreateMainWindow() override {}
    void* GetMainWindowHandler() override { return nullptr; }

    void RegisterManagerModule(IRuntimeModule* module);
    void RegisterManagerModule(IGraphicsManager* mgr);
    void RegisterManagerModule(IMemoryManager* mgr);
    void RegisterManagerModule(IAssetLoader* mgr);
    void RegisterManagerModule(IInputManager* mgr);
    void RegisterManagerModule(ISceneManager* mgr);
    void RegisterManagerModule(IAnimationManager* mgr);
    void RegisterManagerModule(IPhysicsManager* mgr);
    void RegisterManagerModule(IPipelineStateManager* mgr);
    void RegisterManagerModule(IGameLogic* logic);
#ifdef DEBUG
    void RegisterManagerModule(IDebugManager* mgr);
#endif

    // Rebind the borrowed session-manager pointers without lifecycle
    // registration: the runtime root owns module lifecycle, and replaced
    // session modules may reuse freed addresses (duplicate registration
    // would throw). Event/resize routing only borrows these pointers.
    void BindSessionManagers(IGraphicsManager* graphics,
                             IPipelineStateManager* pipelines) {
        m_pGraphicsManager = graphics;
        m_pPipelineStateManager = pipelines;
    }

    IGraphicsManager* GetGraphicsManager() { return m_pGraphicsManager; }
    IMemoryManager* GetMemoryManager() { return m_pMemoryManager; }
    IAssetLoader* GetAssetLoader() { return m_pAssetLoader; }
    IInputManager* GetInputManager() { return m_pInputManager; }
    ISceneManager* GetSceneManager() { return m_pSceneManager; }
    IAnimationManager* GetAnimationManager() { return m_pAnimationManager; }
    IPhysicsManager* GetPhysicsManager() { return m_pPhysicsManager; }
    IPipelineStateManager* GetPipelineStateManager() {
        return m_pPipelineStateManager;
    }
    IGameLogic* GetGameLogic() { return m_pGameLogic; }
    // nullptr unless built with USE_ENGINE_ALLOCATORS and running.
    MemorySystem* GetMemorySystem() { return m_pMemorySystem.get(); }
#ifdef DEBUG
    IDebugManager* GetDebugManager() { return m_pDebugManager; }
#endif

   protected:
    virtual int InitializeGraphicsResources() { return 0; }
    virtual int InitializeGuiPlatform() { return 0; }
    virtual void FinalizeGuiPlatform() {}
    // Drain GPU work before shutdown or failure rollback releases module resources.
    virtual void WaitForGraphicsIdle() {}
    virtual void FinalizeGraphicsResources() {}
    virtual void FinalizePlatformResources() {}

    // Flag if need quit the main loop of the application
    bool m_bQuit = false;
    GfxConfiguration m_Config;
    int m_nArgC = 0;
    char** m_ppArgV = nullptr;

    IGraphicsManager* m_pGraphicsManager = nullptr;
    IMemoryManager* m_pMemoryManager = nullptr;
    IAssetLoader* m_pAssetLoader = nullptr;
    IInputManager* m_pInputManager = nullptr;
    ISceneManager* m_pSceneManager = nullptr;
    IAnimationManager* m_pAnimationManager = nullptr;
    IPhysicsManager* m_pPhysicsManager = nullptr;
    IPipelineStateManager* m_pPipelineStateManager = nullptr;
    IGameLogic* m_pGameLogic = nullptr;
#ifdef DEBUG
    IDebugManager* m_pDebugManager = nullptr;
#endif

   private:
    void ReleaseMemorySystem();
    bool m_GraphicsInitialized = false;
    bool m_GuiPlatformInitialized = false;
    RuntimeModuleGroup runtime_modules;
    std::unique_ptr<MemorySystem> m_pMemorySystem;
};
}  // namespace My
