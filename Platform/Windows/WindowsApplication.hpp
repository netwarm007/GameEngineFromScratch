#include <windows.h>
#include <windowsx.h>
#include <d3d12.h>
#include <DXGI1_4.h>
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

    private:
        void GetHardwareAdapter(IDXGIFactory4* pFactory, IDXGIAdapter1** ppAdapter);

    private:
        IDXGISwapChain3*         m_pSwapChain = nullptr;             // the pointer to the swap chain interface
        ID3D12Device*            m_pDev       = nullptr;             // the pointer to our Direct3D device interface
    };
}

