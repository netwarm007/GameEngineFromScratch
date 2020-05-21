#include <X11/Xlib.h>

#include "CefApplication.hpp"

namespace My {
int XErrorHandlerImpl(Display* display, XErrorEvent* event) {
    LOG(WARNING) << "X error received: "
                 << "type " << event->type << ", "
                 << "serial " << event->serial << ", "
                 << "error_code " << static_cast<int>(event->error_code) << ", "
                 << "request_code " << static_cast<int>(event->request_code)
                 << ", "
                 << "minor_code " << static_cast<int>(event->minor_code);
    return 0;
}

int XIOErrorHandlerImpl(Display* display) { return 0; }
}  // namespace My

using namespace My;

int CefApplication::Initialize() {
    BaseApplication::Initialize();

    CefEnableHighDPISupport();

    void* sandbox_info = NULL;

#if defined(CEF_USE_SANDBOX)
    CefScopedSandboxInfo scoped_sandbox;
    sandbox_info = scoped_sandbox.sandbox_info();
#endif

    CefMainArgs main_args(m_nArgC, m_ppArgV);

    int exit_code = CefExecuteProcess(main_args, NULL, sandbox_info);
    if (exit_code >= 0) {
        return exit_code;
    }

    // Install xlib error handlers so that the application won't be terminated
    // on non-fatal errors.
    XSetErrorHandler(XErrorHandlerImpl);
    XSetIOErrorHandler(XIOErrorHandlerImpl);

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
