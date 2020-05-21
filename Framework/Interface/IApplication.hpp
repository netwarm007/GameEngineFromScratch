#pragma once
#include "GfxConfiguration.hpp"
#include "IRuntimeModule.hpp"

namespace My {
_Interface_ IApplication : _inherits_ IRuntimeModule {
   public:
    int Initialize() override = 0;
    void Finalize() override = 0;
    // One cycle of the main loop
    void Tick() override = 0;

    virtual void SetCommandLineParameters(int argc, char** argv) = 0;
    [[nodiscard]] virtual int GetCommandLineArgumentsCount() const = 0;
    [[nodiscard]] virtual const char* GetCommandLineArgument(int index)
        const = 0;

    [[nodiscard]] virtual bool IsQuit() const = 0;
    virtual void RequestQuit() = 0;

    virtual void* GetMainWindowHandler() = 0;

    [[nodiscard]] virtual const GfxConfiguration& GetConfiguration() const = 0;

   protected:
    virtual void CreateMainWindow() = 0;
};

extern IApplication* g_pApp;
}  // namespace My
