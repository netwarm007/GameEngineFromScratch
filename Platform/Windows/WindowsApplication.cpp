// include the basic windows header file
#include "WindowsApplication.hpp"
#include <tchar.h>
#include <objbase.h>

using namespace My;

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 32, 0, 0, 960, 540, L"Game Engine From Scratch (Windows)");
    WindowsApplication  g_App(config);
    IApplication*       g_pApp = &g_App;
}

void My::WindowsApplication::GetHardwareAdapter(IDXGIFactory4* pFactory, IDXGIAdapter1** ppAdapter)
{
    IDXGIAdapter1* pAdapter = nullptr;
    *ppAdapter = nullptr;

 	for (UINT adapterIndex = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &pAdapter); adapterIndex++)
 	{
	   DXGI_ADAPTER_DESC1 desc;
	   pAdapter->GetDesc1(&desc);

	   if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
	   {
		   // Don't select the Basic Render Driver adapter.
		   continue;
	   }

	   // Check to see if the adapter supports Direct3D 12, but don't create the
	   // actual device yet.
	   if (SUCCEEDED(D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr)))
	   {
		   break;
	   }
	}

 	*ppAdapter = pAdapter;
}

int My::WindowsApplication::Initialize()
{
    int result;

    result = BaseApplication::Initialize();

    if (result != 0)
        exit(result);

    // get the HINSTANCE of the Console Program
    HINSTANCE hInstance = GetModuleHandle(NULL);

    // the handle for the window, filled by a function
    HWND hWnd;
    // this struct holds information for the window class
    WNDCLASSEX wc;

    // clear out the window class for use
    ZeroMemory(&wc, sizeof(WNDCLASSEX));

    // fill in the struct with the needed information
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)COLOR_WINDOW;
    wc.lpszClassName = _T("GameEngineFromScratch");

    // register the window class
    RegisterClassEx(&wc);

    // create the window and use the result as the handle
    hWnd = CreateWindowExW(0,
                          L"GameEngineFromScratch",      // name of the window class
                          m_Config.appName,             // title of the window
                          WS_OVERLAPPEDWINDOW,              // window style
                          CW_USEDEFAULT,                    // x-position of the window
                          CW_USEDEFAULT,                    // y-position of the window
                          m_Config.screenWidth,             // width of the window
                          m_Config.screenHeight,            // height of the window
                          NULL,                             // we have no parent window, NULL
                          NULL,                             // we aren't using menus, NULL
                          hInstance,                        // application handle
                          NULL);                            // used with multiple windows, NULL

    // display the window on the screen
    ShowWindow(hWnd, SW_SHOW);

#if defined(_DEBUG)
	// Enable the D3D12 debug layer.
	{
		ID3D12Debug* pDebugController;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pDebugController))))
		{
			pDebugController->EnableDebugLayer();
		}
		pDebugController->Release();
	}
#endif

	IDXGIFactory4* pFactory;
	if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&pFactory)))) {
		return -1;
	}

	IDXGIAdapter1* pHardwareAdapter;
	GetHardwareAdapter(pFactory, &pHardwareAdapter);

	if (FAILED(D3D12CreateDevice(pHardwareAdapter,
		D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_pDev)))) {

		IDXGIAdapter* pWarpAdapter;
		if (SUCCEEDED(pFactory->EnumWarpAdapter(IID_PPV_ARGS(&pWarpAdapter)))) {
			if(FAILED(D3D12CreateDevice(pWarpAdapter, D3D_FEATURE_LEVEL_11_0,
				IID_PPV_ARGS(&m_pDev)))) {
				result = -1;
			}
		} else {
			result = -1;
		}

	}

	pFactory->Release();

    return result;
}

void My::WindowsApplication::Finalize()
{
}

void My::WindowsApplication::Tick()
{
    // this struct holds Windows event messages
    MSG msg;

    // we use PeekMessage instead of GetMessage here
    // because we should not block the thread at anywhere
    // except the engine execution driver module 
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        // translate keystroke messages into the right format
        TranslateMessage(&msg);

        // send the message to the WindowProc function
        DispatchMessage(&msg); 
    }
}

// this is the main message handler for the program
LRESULT CALLBACK My::WindowsApplication::WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    // sort through and find what code to run for the message given
    switch(message)
    {
	case WM_PAINT:
        // we will replace this part with Rendering Module
	    {
	    } break;

        // this message is read when the window is closed
    case WM_DESTROY:
        {
            // close the application entirely
            PostQuitMessage(0);
            BaseApplication::m_bQuit = true;
            return 0;
        }
    }

    // Handle any messages the switch statement didn't
    return DefWindowProc (hWnd, message, wParam, lParam);
}


