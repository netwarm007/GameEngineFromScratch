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

   protected:
    void CreateMainWindow() override;

   private:
    // the WindowProc function prototype
    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam,
                                       LPARAM lParam);

   protected:
    HWND m_hWnd;
    HDC m_hDc;
    bool m_bInLeftDrag = false;
    bool m_bInRightDrag = false;
    int m_iPreviousX = 0;
    int m_iPreviousY = 0;
};
}  // namespace My
