#include "WindowsApplication.hpp"

#include <tchar.h>

#include "InputManager.hpp"
#include "imgui/examples/imgui_impl_win32.h"

using namespace My;

void WindowsApplication::CreateMainWindow() {
    // get the HINSTANCE of the Console Program
    HINSTANCE hInstance = GetModuleHandle(NULL);

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

    int height_adjust =
        (GetSystemMetrics(SM_CYFRAME) + GetSystemMetrics(SM_CYCAPTION) +
         GetSystemMetrics(SM_CXPADDEDBORDER));
    int width_adjust =
        (GetSystemMetrics(SM_CXFRAME) + GetSystemMetrics(SM_CXPADDEDBORDER));

    // create the window and use the result as the handle
    m_hWnd = CreateWindowEx(
        0,
        _T("GameEngineFromScratch"),            // name of the window class
        m_Config.appName,                       // title of the window
        WS_OVERLAPPEDWINDOW,                    // window style
        CW_USEDEFAULT,                          // x-position of the window
        CW_USEDEFAULT,                          // y-position of the window
        m_Config.screenWidth + width_adjust,    // width of the window
        m_Config.screenHeight + height_adjust,  // height of the window
        NULL,       // we have no parent window, NULL
        NULL,       // we aren't using menus, NULL
        hInstance,  // application handle
        this);      // pass pointer to current object

    m_hDc = GetDC(m_hWnd);

    // display the window on the screen
    ShowWindow(m_hWnd, SW_SHOW);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    [[maybe_unused]] ImGuiIO& io = ImGui::GetIO();

    ImGui_ImplWin32_Init(m_hWnd);
    ImGui_ImplWin32_EnableDpiAwareness();

    ImGui::StyleColorsDark();
}

void WindowsApplication::Finalize() {
    // Finalize ImGui
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    ReleaseDC(m_hWnd, m_hDc);

    BaseApplication::Finalize();
}

void WindowsApplication::Tick() {
    BaseApplication::Tick();

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

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd,
                                                             UINT msg,
                                                             WPARAM wParam,
                                                             LPARAM lParam);

// this is the main message handler for the program
LRESULT CALLBACK WindowsApplication::WindowProc(HWND hWnd, UINT message,
                                                WPARAM wParam, LPARAM lParam) {
    LRESULT result = 0;

    WindowsApplication* pThis;
    if (message == WM_NCCREATE) {
        pThis = static_cast<WindowsApplication*>(
            reinterpret_cast<CREATESTRUCT*>(lParam)->lpCreateParams);

        SetLastError(0);
        if (!SetWindowLongPtr(hWnd, GWLP_USERDATA,
                              reinterpret_cast<LONG_PTR>(pThis))) {
            if (GetLastError() != 0) return FALSE;
        }
    } else {
        pThis = reinterpret_cast<WindowsApplication*>(
            GetWindowLongPtr(hWnd, GWLP_USERDATA));
    }

    // ImGui message handler
    result = ImGui_ImplWin32_WndProcHandler(hWnd, message, wParam, lParam);

    // sort through and find what code to run for the message given
    switch (message) {
        case WM_CHAR: {
            g_pInputManager->AsciiKeyDown(static_cast<char>(wParam));
        } break;
        case WM_KEYUP: {
            switch (wParam) {
                case VK_LEFT:
                    g_pInputManager->LeftArrowKeyUp();
                    break;
                case VK_RIGHT:
                    g_pInputManager->RightArrowKeyUp();
                    break;
                case VK_UP:
                    g_pInputManager->UpArrowKeyUp();
                    break;
                case VK_DOWN:
                    g_pInputManager->DownArrowKeyUp();
                    break;

                default:
                    break;
            }
        } break;
        case WM_KEYDOWN: {
            switch (wParam) {
                case VK_LEFT:
                    g_pInputManager->LeftArrowKeyDown();
                    break;
                case VK_RIGHT:
                    g_pInputManager->RightArrowKeyDown();
                    break;
                case VK_UP:
                    g_pInputManager->UpArrowKeyDown();
                    break;
                case VK_DOWN:
                    g_pInputManager->DownArrowKeyDown();
                    break;

                default:
                    break;
            }
        } break;
        case WM_LBUTTONDOWN: {
            g_pInputManager->LeftMouseButtonDown();
            pThis->m_bInLeftDrag = true;
            pThis->m_iPreviousX = GET_X_LPARAM(lParam);
            pThis->m_iPreviousY = GET_Y_LPARAM(lParam);
        } break;
        case WM_LBUTTONUP: {
            g_pInputManager->LeftMouseButtonUp();
            pThis->m_bInLeftDrag = false;
        } break;
        case WM_RBUTTONDOWN: {
            g_pInputManager->RightMouseButtonDown();
            pThis->m_bInRightDrag = true;
            pThis->m_iPreviousX = GET_X_LPARAM(lParam);
            pThis->m_iPreviousY = GET_Y_LPARAM(lParam);
        } break;
        case WM_RBUTTONUP: {
            g_pInputManager->RightMouseButtonUp();
            pThis->m_bInRightDrag = false;
        } break;
        case WM_MOUSEMOVE: {
            int pos_x = GET_X_LPARAM(lParam);
            int pos_y = GET_Y_LPARAM(lParam);
            if (pThis->m_bInLeftDrag) {
                g_pInputManager->LeftMouseDrag(pos_x - pThis->m_iPreviousX,
                                               pos_y - pThis->m_iPreviousY);
            }
            else if (pThis->m_bInRightDrag) {
                g_pInputManager->RightMouseDrag(pos_x - pThis->m_iPreviousX,
                                               pos_y - pThis->m_iPreviousY);
            }
        } break;
        // this message is read when the window is closed
        case WM_DESTROY: {
            // close the application entirely
            PostQuitMessage(0);
            m_bQuit = true;
        } break;
        default:
            // Handle any messages the switch statement didn't
            result = DefWindowProc(hWnd, message, wParam, lParam);
    }

    return result;
}
