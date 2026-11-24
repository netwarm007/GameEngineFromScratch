#include "RhiFrameDriver.hpp"

#include <iostream>

#include "Framework/Interface/IAssetReader.hpp"
#include "Framework/Interface/ISceneManager.hpp"
#include "Framework/Manager/BaseApplication.hpp"
#include "Framework/Manager/InputManager.hpp"
#include "Framework/Manager/RuntimeRoot.hpp"
#include "RhiComposition.hpp"

using namespace My;

RhiFrameDriver::RhiFrameDriver(
    RhiSwitchController& controller, RuntimeRoot& root,
    RhiStartupComposition& composition, BaseApplication& app, InputManager& input,
    ISceneManager& scene, IAssetReader& assets, IGraphicsManager*& graphicsManager,
    IPipelineStateManager*& pipelineStateManager,
    std::function<MemorySystem*()> memoryProvider,
    std::vector<RhiSwitchStep> schedule, int reconstructCycles, Hooks hooks)
    : m_Controller(controller),
      m_Root(root),
      m_Composition(composition),
      m_App(app),
      m_Input(input),
      m_Scene(scene),
      m_Assets(assets),
      m_GraphicsManager(graphicsManager),
      m_PipelineStateManager(pipelineStateManager),
      m_MemoryProvider(std::move(memoryProvider)),
      m_Schedule(std::move(schedule)),
      m_ReconstructCycles(reconstructCycles),
      m_Hooks(std::move(hooks)) {}

void RhiFrameDriver::RebindSession() {
    m_GraphicsManager = m_Composition.graphicsManager.get();
    m_PipelineStateManager = m_Composition.pipelineStateManager.get();
}

int RhiFrameDriver::Tick() {
    m_Root.Tick();
    ++m_FrameCount;
    while (m_NextSwitch < m_Schedule.size() &&
           m_FrameCount >= m_Schedule[m_NextSwitch].frame) {
        const auto target = m_Schedule[m_NextSwitch].backend;
        const auto result = m_Controller.Request(target);
        std::cout << "RHI switch request " << RhiBackendName(target) << ": "
                  << RhiSwitchRequestResultName(result) << '\n';
        ++m_NextSwitch;
    }
    m_Controller.ProcessFrameBoundary();
    int ret = 0;
    if (m_Controller.ExitRequested()) {
        std::cerr << "RHI switch: " << m_Controller.LastError() << '\n';
        m_App.RequestQuit();
        ret = 1;
    }
    if (m_Controller.ConsumeFramePlanDirty()) {
        // Reconcile held input: keys physically released while the window was
        // gone must not stay pressed in the new session.
        m_Input.Initialize();
        RebindSession();
        if (m_Hooks.onSessionRebound) m_Hooks.onSessionRebound();
        if (m_Hooks.applyFramePlan) m_Hooks.applyFramePlan(m_SimulationPaused);
    }
    if (m_Controller.IsSimulationPaused() != m_SimulationPaused) {
        m_SimulationPaused = m_Controller.IsSimulationPaused();
        if (m_Hooks.applyFramePlan) m_Hooks.applyFramePlan(m_SimulationPaused);
    }
    if (m_ReconstructCycles > 0) {
        const int cycles = m_ReconstructCycles;
        m_ReconstructCycles = 0;
        for (int i = 0; i < cycles; ++i) {
            if (ReconstructRhiSession(m_Root, m_Composition, m_Scene, m_Assets,
                                      m_MemoryProvider) != 0) {
                std::cerr << "RHI session reconstruction failed\n";
                m_App.RequestQuit();
                break;
            }
            RebindSession();
            if (m_Hooks.onSessionRebound) m_Hooks.onSessionRebound();
            if (m_Hooks.applyFramePlan) m_Hooks.applyFramePlan(false);
            std::cout << "RHI session reconstructed (cycle " << i + 1 << ")\n";
        }
    }
    return ret;
}