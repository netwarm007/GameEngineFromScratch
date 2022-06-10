#pragma once
#include "GfxConfiguration.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ IApplication {
   public:
    virtual ~IApplication() = default;
    virtual void SetCommandLineParameters(int argc, char** argv) = 0;
    [[nodiscard]] virtual int GetCommandLineArgumentsCount() const = 0;
    [[nodiscard]] virtual const char* GetCommandLineArgument(int index)
        const = 0;

    [[nodiscard]] virtual bool IsQuit() const = 0;
    virtual void RequestQuit() = 0;

    virtual void CreateMainWindow() = 0;
    virtual void* GetMainWindowHandler() = 0;

    [[nodiscard]] virtual const GfxConfiguration& GetConfiguration() const = 0;

    virtual void GetFramebufferSize(int& width, int& height) = 0;
};
}  // namespace My
