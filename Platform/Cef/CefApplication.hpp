#pragma once
#include "BaseApplication.hpp"
#include "cef_app.h"

namespace My {
class CefApplication : public BaseApplication,
                       public CefApp,
                       public CefBrowserProcessHandler {
   public:
    explicit CefApplication(GfxConfiguration& config)
        : BaseApplication(config) {}

    int Initialize() override;
    void Finalize() override;
    // One cycle of the main loop
    void Tick() override;

    void* GetMainWindowHandler() override { return this; }

    // CefApp methods:
    CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() override {
        return this;
    }

    // CefBrowserProcessHandler methods:
    void OnContextInitialized() override;

   private:
    // Include the default reference counting implementation.
    IMPLEMENT_REFCOUNTING(CefApplication);
};
}  // namespace My
