#pragma once
#include "cef_app.h"
#include "BaseApplication.hpp"

namespace My {
    class CefApplication : public BaseApplication, public CefApp, public CefBrowserProcessHandler
    {
    public:
        CefApplication(GfxConfiguration& config)
            : BaseApplication(config) {}

        virtual int Initialize();
        virtual void Finalize();
        // One cycle of the main loop
        virtual void Tick();

        // CefApp methods:
        virtual CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() override
        {
            return this;
        }

        // CefBrowserProcessHandler methods:
        virtual void OnContextInitialized() override;

    private:
        // Include the default reference counting implementation.
        IMPLEMENT_REFCOUNTING(CefApplication);
    };
}
