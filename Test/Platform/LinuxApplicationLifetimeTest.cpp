#include <iostream>

#include "OpenGLApplication.hpp"

using namespace My;

namespace {
class ContextCheckingModule : public IMemoryManager {
   public:
    explicit ContextCheckingModule(OpenGLApplication& app) : m_App(app) {}

    int Initialize() override { return 0; }
    void Finalize() override {
        valid = m_App.GetMainWindowHandler() != nullptr &&
                glXGetCurrentContext() != nullptr;
        finalized = true;
    }
    void Tick() override {}
    void* AllocatePage(size_t) override { return nullptr; }
    void FreePage(void*) override {}

    bool finalized = false;
    bool valid = false;

   private:
    OpenGLApplication& m_App;
};
}  // namespace

int main() {
    GfxConfiguration config;
    config.screenWidth = 64;
    config.screenHeight = 64;
    OpenGLApplication app(config);
    ContextCheckingModule module(app);
    app.CreateMainWindow();
    app.RegisterManagerModule(&module);
    const int result = app.Initialize();
    app.Finalize();
    if (result != 0 || !module.finalized || !module.valid ||
        app.GetMainWindowHandler() != nullptr ||
        glXGetCurrentContext() != nullptr) {
        std::cerr << "Linux application did not finalize modules with a live "
                     "context before releasing the backend and window\n";
        return 1;
    }
    app.Finalize();
    return 0;
}
