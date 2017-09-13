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

        // the WindowProc function prototype
        static LRESULT CALLBACK WindowProc(HWND hWnd,
                         UINT message,
                         WPARAM wParam,
                         LPARAM lParam);

        inline HWND GetMainWindow() { return m_hWnd; };

    private:
        HWND m_hWnd;
    };
}

