#pragma once

#include <functional>

#include "IRenderBackend.hpp"
#include "portable.hpp"

OBJC_CLASS(MetalView);

namespace My {
class IApplication;
class IPlatformHost;
class IInputManager;

// Session-owned Metal backend lifecycle against a persistent Cocoa host
// (M4.1 detachment). Creates the Metal drawable on the host window; the
// Metal RHI itself is owned by the graphics manager, and the drawable is
// window-owned, so the native teardown hooks have no state to release.
class MetalBackend : _implements_ IRenderBackend {
   public:
    MetalBackend(IApplication& application, IPlatformHost& platform,
                 std::function<IInputManager*()> inputProvider);

    // The installed Metal drawable; nullptr before graphics resources exist
    // or when the content view is not Metal.
    MetalView* GetMetalView();

    void CreateMainWindow() override;
    int InitializeGraphicsResources() override;
    void WaitForGraphicsIdle() override {}
    void FinalizeGraphicsResources() override {}
    int InitializeGuiPlatform() override;
    void FinalizeGuiPlatform() override;

   private:
    IApplication& m_Application;
    IPlatformHost& m_Platform;
    std::function<IInputManager*()> m_InputProvider;
};
}  // namespace My