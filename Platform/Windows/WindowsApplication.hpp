#pragma once
// include the basic windows header file
#include <windows.h>
#include <windowsx.h>

#include "BaseApplication.hpp"
#include "IPlatformHost.hpp"

namespace My {
class WindowsApplication : public BaseApplication, _implements_ IPlatformHost {
   public:
    using BaseApplication::BaseApplication;

    // One cycle of the main loop
    void Tick() override;

    void ProcessEvents() override;

    void* GetMainWindowHandler() override { return m_hWnd; };

    void CreateMainWindow() override;

    void GetFramebufferSize(uint32_t& width, uint32_t& height) override;

    int InitializeGuiPlatform() override;
    void FinalizeGuiPlatform() override;
    void FinalizePlatformResources() override;

    void SetActiveBackend(IRenderBackend* backend) override {
        m_pActiveBackend = backend;
    }

   protected:
    virtual void onWindowResize(int new_width, int new_height);

   private:
    // the WindowProc function prototype
    static LRESULT CALLBACK m_fWindowProc(HWND hWnd, UINT message,
                                          WPARAM wParam, LPARAM lParam);

   protected:
    HINSTANCE m_hInstance = NULL;
    HWND m_hWnd = NULL;
    HDC m_hDc = NULL;
    IRenderBackend* m_pActiveBackend = nullptr;
    // Logical window geometry survives reconstruction/switching (M4.3);
    // native handles are never preserved.
    bool m_HasPreservedGeometry = false;
    RECT m_PreservedRect{};
    bool m_bInLeftDrag = false;
    bool m_bInRightDrag = false;
    int m_iPreviousX = 0;
    int m_iPreviousY = 0;
};
}  // namespace My
