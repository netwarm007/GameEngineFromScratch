#pragma once
#include "cef_app.h"
#include "BaseApplication.hpp"

namespace My {
    class CefApplication : public BaseApplication, public CefApp, public CefBrowserProcessHandler
    {
    public:
        CefApplication(GfxConfiguration& config)
            : BaseApplication(config) {}

        int Initialize() override;
        void Finalize() override;
        // One cycle of the main loop
        void Tick() override;

        void* GetMainWindowHandler() override { return this; };

        // CefApp methods:
        CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() override
        {
            return this;
        }

        // CefBrowserProcessHandler methods:
        void OnContextInitialized() override;

    private:
        // Include the default reference counting implementation.
        IMPLEMENT_REFCOUNTING(CefApplication);
    };
}
