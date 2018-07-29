#pragma once
// include the basic windows header file
#include <windows.h>
#include <windowsx.h>
#include "BaseApplication.hpp"

namespace My {
    class WindowsApplication : public BaseApplication
    {
    public:
        WindowsApplication(GfxConfiguration& config)
            : BaseApplication(config) {};

        virtual int Initialize();
        virtual void Finalize();
        // One cycle of the main loop
        virtual void Tick();

        HWND GetMainWindow() { return m_hWnd; };

    protected:
        void CreateMainWindow();

    private:
        // the WindowProc function prototype
        static LRESULT CALLBACK WindowProc(HWND hWnd,
                         UINT message,
                         WPARAM wParam,
                         LPARAM lParam);

    protected:
        HWND m_hWnd;
        HDC  m_hDc;
        bool m_bInDrag = false;
        int  m_iPreviousX = 0;
        int  m_iPreviousY = 0;
    };
}

