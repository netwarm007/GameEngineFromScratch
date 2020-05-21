#include <windows.h>

#include "CefApplication.hpp"

using namespace My;

int CefApplication::Initialize() {
    BaseApplication::Initialize();

    CefEnableHighDPISupport();

    void* sandbox_info = NULL;

#if defined(CEF_USE_SANDBOX)
    CefScopedSandboxInfo scoped_sandbox;
    sandbox_info = scoped_sandbox.sandbox_info();
#endif

    HINSTANCE hInstance = GetModuleHandle(NULL);
    CefMainArgs main_args(hInstance);

    int exit_code = CefExecuteProcess(main_args, NULL, sandbox_info);
    if (exit_code >= 0) {
        return exit_code;
    }

    CefSettings settings;

#if !defined(CEF_USE_SANDBOX)
    settings.no_sandbox = true;
#endif

    CefRefPtr<CefApplication> app(this);

    CefInitialize(main_args, settings, app.get(), sandbox_info);

    return 0;
}

void CefApplication::Finalize() { CefShutdown(); }

void CefApplication::Tick() { CefDoMessageLoopWork(); }
