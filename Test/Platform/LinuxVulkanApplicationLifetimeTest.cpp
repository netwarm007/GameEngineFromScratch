#include <iostream>

#include "VulkanApplication.hpp"

using namespace My;

namespace {
class CountingModule : public IMemoryManager {
   public:
    int Initialize() override {
        ++initializeCount;
        return 0;
    }
    void Finalize() override { ++finalizeCount; }
    void Tick() override { ++tickCount; }
    void* AllocatePage(size_t) override { return nullptr; }
    void FreePage(void*) override {}

    int initializeCount = 0;
    int finalizeCount = 0;
    int tickCount = 0;
};
}  // namespace

// The test environment hides every Vulkan driver, so native initialization
// fails after the window exists. The application must report the failure by
// error code, leave the window for the caller's Finalize, and never start modules.
int main() {
    GfxConfiguration config;
    config.screenWidth = config.screenHeight = 64;
    VulkanApplication app(config);
    CountingModule module;

    app.CreateMainWindow();
    app.RegisterManagerModule(&module);
    const bool windowCreated = app.GetMainWindowHandler() != nullptr;
    int result = 0;
    bool escaped = false;
    try {
        result = app.Initialize();
    } catch (const std::exception& error) {
        std::cerr << "Vulkan initialization failure escaped the application: "
                  << error.what() << "\n";
        escaped = true;
    }
    const bool windowAlive = app.GetMainWindowHandler() != nullptr;
    app.Tick();
    app.Finalize();
    app.Finalize();

    if (!windowCreated || escaped || result == 0 || !windowAlive ||
        module.initializeCount != 0 || module.finalizeCount != 0 ||
        module.tickCount != 0 || app.GetRHI().GetDevice() ||
        app.GetMainWindowHandler() != nullptr) {
        std::cerr << "Vulkan application did not preserve the initialization "
                     "failure and cleanup contract: result=" << result
                  << " escaped=" << escaped << " window=" << windowCreated
                  << "/" << windowAlive << " modules="
                  << module.initializeCount << "/" << module.finalizeCount
                  << "/" << module.tickCount << "\n";
        return 1;
    }
    return 0;
}
