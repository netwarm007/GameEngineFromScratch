#pragma once
// include the basic windows header file
#include <windows.h>
#include <windowsx.h>

#include "BaseApplication.hpp"

namespace My {
class WindowsApplication : public BaseApplication {
   public:
    using BaseApplication::BaseApplication;

    void Finalize() override;
    // One cycle of the main loop
    void Tick() override;

    void* GetMainWindowHandler() override { return m_hWnd; };

    void CreateMainWindow() override;

    void GetFramebufferSize(int& width, int& height) override;

   protected:
    virtual void onWindowResize(int new_width, int new_height) {}

   private:
    // the WindowProc function prototype
    static LRESULT CALLBACK m_fWindowProc(HWND hWnd, UINT message,
                                          WPARAM wParam, LPARAM lParam);

   protected:
    HINSTANCE m_hInstance = NULL;
    HWND m_hWnd = NULL;
    HDC m_hDc = NULL;
    bool m_bInLeftDrag = false;
    bool m_bInRightDrag = false;
    int m_iPreviousX = 0;
    int m_iPreviousY = 0;
};
}  // namespace My
