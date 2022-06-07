#pragma once
#include <vector>
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

namespace My {
class BaseApplication : _implements_ IApplication {
   public:
    BaseApplication() = default;
    explicit BaseApplication(GfxConfiguration& cfg) : m_Config(cfg) {}
    ~BaseApplication() override = default;
    virtual int Initialize();
    virtual void Finalize();
    // One cycle of the main loop
    virtual void Tick();

    void SetCommandLineParameters(int argc, char** argv) override;
    [[nodiscard]] int GetCommandLineArgumentsCount() const override;
    [[nodiscard]] const char* GetCommandLineArgument(int index) const override;

    [[nodiscard]] bool IsQuit() const override;
    void RequestQuit() override { m_bQuit = true; }

    [[nodiscard]] inline const GfxConfiguration& GetConfiguration()
        const override {
        return m_Config;
    }

    void CreateMainWindow() override {}
    void* GetMainWindowHandler() override { return nullptr; }

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
#ifdef DEBUG 
    IDebugManager* GetDebugManager() { return m_pDebugManager; }
#endif

   protected:
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
    std::vector<IRuntimeModule*> runtime_modules;
};
}  // namespace My
