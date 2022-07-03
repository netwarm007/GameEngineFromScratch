#include "WindowsApplication.hpp"

#include <tchar.h>

#include "InputManager.hpp"
#include "imgui_impl_win32.h"

using namespace My;

void WindowsApplication::CreateMainWindow() {
    // get the HINSTANCE of the Console Program
    m_hInstance = GetModuleHandle(NULL);

    // this struct holds information for the window class
    WNDCLASSEX wc;

    // clear out the window class for use
    ZeroMemory(&wc, sizeof(WNDCLASSEX));

    // fill in the struct with the needed information
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = m_fWindowProc;
    wc.hInstance = m_hInstance;
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
        NULL,         // we have no parent window, NULL
        NULL,         // we aren't using menus, NULL
        m_hInstance,  // application handle
        this);        // pass pointer to current object

    m_hDc = GetDC(m_hWnd);

    // display the window on the screen
    ShowWindow(m_hWnd, SW_SHOW);
    ImGui_ImplWin32_Init(m_hWnd);
    ImGui_ImplWin32_EnableDpiAwareness();
}

void WindowsApplication::Finalize() {
    BaseApplication::Finalize();

    // Finalize ImGui
    ImGui_ImplWin32_Shutdown();

    ReleaseDC(m_hWnd, m_hDc);
}

void WindowsApplication::Tick() {
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

    ImGui_ImplWin32_NewFrame();
    BaseApplication::Tick();
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd,
                                                             UINT msg,
                                                             WPARAM wParam,
                                                             LPARAM lParam);

// this is the main message handler for the program
LRESULT CALLBACK WindowsApplication::m_fWindowProc(HWND hWnd, UINT message,
                                                   WPARAM wParam,
                                                   LPARAM lParam) {
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
        case WM_SIZE:
            if (pThis) {
                pThis->onWindowResize(LOWORD(lParam), HIWORD(lParam));
            }
            break;
        case WM_CHAR:
            if (pThis && pThis->m_pInputManager) {
                pThis->m_pInputManager->AsciiKeyDown(static_cast<char>(wParam));
            }
            break;
        case WM_KEYUP:
            if (pThis && pThis->m_pInputManager) {
                switch (wParam) {
                    case VK_LEFT:
                        pThis->m_pInputManager->LeftArrowKeyUp();
                        break;
                    case VK_RIGHT:
                        pThis->m_pInputManager->RightArrowKeyUp();
                        break;
                    case VK_UP:
                        pThis->m_pInputManager->UpArrowKeyUp();
                        break;
                    case VK_DOWN:
                        pThis->m_pInputManager->DownArrowKeyUp();
                        break;

                    default:
                        break;
                }
            }
            break;
        case WM_KEYDOWN:
            if (pThis && pThis->m_pInputManager) {
                switch (wParam) {
                    case VK_LEFT:
                        pThis->m_pInputManager->LeftArrowKeyDown();
                        break;
                    case VK_RIGHT:
                        pThis->m_pInputManager->RightArrowKeyDown();
                        break;
                    case VK_UP:
                        pThis->m_pInputManager->UpArrowKeyDown();
                        break;
                    case VK_DOWN:
                        pThis->m_pInputManager->DownArrowKeyDown();
                        break;

                    default:
                        break;
                }
            }
            break;
        case WM_LBUTTONDOWN:
            if (pThis && pThis->m_pInputManager) {
                pThis->m_pInputManager->LeftMouseButtonDown();
                pThis->m_bInLeftDrag = true;
                pThis->m_iPreviousX = GET_X_LPARAM(lParam);
                pThis->m_iPreviousY = GET_Y_LPARAM(lParam);
            }
            break;
        case WM_LBUTTONUP:
            if (pThis && pThis->m_pInputManager) {
                pThis->m_pInputManager->LeftMouseButtonUp();
                pThis->m_bInLeftDrag = false;
            }
            break;
        case WM_RBUTTONDOWN:
            if (pThis && pThis->m_pInputManager) {
                pThis->m_pInputManager->RightMouseButtonDown();
                pThis->m_bInRightDrag = true;
                pThis->m_iPreviousX = GET_X_LPARAM(lParam);
                pThis->m_iPreviousY = GET_Y_LPARAM(lParam);
            }
            break;
        case WM_RBUTTONUP:
            if (pThis && pThis->m_pInputManager) {
                pThis->m_pInputManager->RightMouseButtonUp();
                pThis->m_bInRightDrag = false;
            }
            break;
        case WM_MOUSEMOVE:
            if (pThis && pThis->m_pInputManager) {
                int pos_x = GET_X_LPARAM(lParam);
                int pos_y = GET_Y_LPARAM(lParam);
                if (pThis->m_bInLeftDrag) {
                    pThis->m_pInputManager->LeftMouseDrag(
                        pos_x - pThis->m_iPreviousX,
                        pos_y - pThis->m_iPreviousY);
                } else if (pThis->m_bInRightDrag) {
                    pThis->m_pInputManager->RightMouseDrag(
                        pos_x - pThis->m_iPreviousX,
                        pos_y - pThis->m_iPreviousY);
                }
            }
            break;
        case WM_MOUSEWHEEL:
            if (pThis && pThis->m_pInputManager) {
                auto fwKeys = GET_KEYSTATE_WPARAM(wParam);
                auto yDelta = GET_WHEEL_DELTA_WPARAM(wParam) * WHEEL_DELTA;
                pThis->m_pInputManager->RightMouseDrag(0, yDelta);
            }
            break;
        case WM_MOUSEHWHEEL:
            if (pThis && pThis->m_pInputManager) {
                auto fwKeys = GET_KEYSTATE_WPARAM(wParam);
                auto xDelta = GET_WHEEL_DELTA_WPARAM(wParam) * WHEEL_DELTA;
                pThis->m_pInputManager->RightMouseDrag(xDelta, 0);
            }
            break;
        // this message is read when the window is closed
        case WM_CLOSE: {
            // close the application entirely
            if (pThis) {
                pThis->m_bQuit = true;
            }

            return 0;
        } break;
        default:;
    }
    result = DefWindowProc(hWnd, message, wParam, lParam);

    return result;
}

void WindowsApplication::GetFramebufferSize(int& width, int& height) {
    RECT rect;
    GetClientRect(m_hWnd, &rect);

    width = rect.right - rect.left;
    height = rect.bottom - rect.top;
}
