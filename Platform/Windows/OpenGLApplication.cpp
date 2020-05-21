#include "OpenGLApplication.hpp"

#include <stdio.h>
#include <tchar.h>

#include "glad/glad_wgl.h"

using namespace My;

static LRESULT CALLBACK TmpWndProc(HWND hWnd, UINT uiMsg, WPARAM wParam,
                                   LPARAM lParam) {
    switch (uiMsg) {
        case WM_CLOSE:
            PostQuitMessage(0);
            break;

        default:
            return DefWindowProc(hWnd, uiMsg, wParam, lParam);
    }

    return 0;
}

int OpenGLApplication::Initialize() {
    int result;
    auto colorBits =
        m_Config.redBits + m_Config.greenBits +
        m_Config
            .blueBits;  // note on windows this does not include alpha bitplane

    DWORD Style = WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
    WNDCLASSEX WndClassEx;
    memset(&WndClassEx, 0, sizeof(WNDCLASSEX));

    HINSTANCE hInstance = GetModuleHandle(NULL);

    WndClassEx.cbSize = sizeof(WNDCLASSEX);
    WndClassEx.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
    WndClassEx.lpfnWndProc = TmpWndProc;
    WndClassEx.hInstance = hInstance;
    WndClassEx.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    WndClassEx.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
    WndClassEx.hCursor = LoadCursor(NULL, IDC_ARROW);
    WndClassEx.lpszClassName = _T("InitWindow");

    RegisterClassEx(&WndClassEx);
    HWND TemphWnd = CreateWindowEx(WS_EX_APPWINDOW, WndClassEx.lpszClassName,
                                   _T("InitWindow"), Style, 0, 0, CW_USEDEFAULT,
                                   CW_USEDEFAULT, NULL, NULL, hInstance, NULL);

    memset(&m_pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));
    m_pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    m_pfd.nVersion = 1;
    m_pfd.dwFlags = PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW;
    m_pfd.iPixelType = PFD_TYPE_RGBA;
    m_pfd.cColorBits = colorBits;
    m_pfd.cRedBits = m_Config.redBits;
    m_pfd.cGreenBits = m_Config.greenBits;
    m_pfd.cBlueBits = m_Config.blueBits;
    m_pfd.cAlphaBits = m_Config.alphaBits;
    m_pfd.cDepthBits = m_Config.depthBits;
    m_pfd.cStencilBits = m_Config.stencilBits;
    m_pfd.iLayerType = PFD_MAIN_PLANE;

    HDC TemphDC = GetDC(TemphWnd);
    // Set a temporary default pixel format.
    m_nPixelFormat = ChoosePixelFormat(TemphDC, &m_pfd);
    if (m_nPixelFormat == 0) return -1;

    result = SetPixelFormat(TemphDC, m_nPixelFormat, &m_pfd);
    if (result != 1) {
        return result;
    }

    // Create a temporary rendering context.
    m_RenderContext = wglCreateContext(TemphDC);
    if (!m_RenderContext) {
        printf("wglCreateContext failed!\n");
        return -1;
    }

    // Set the temporary rendering context as the current rendering context for
    // this window.
    result = wglMakeCurrent(TemphDC, m_RenderContext);
    if (result != 1) {
        return result;
    }

    if (!gladLoadWGL(TemphDC)) {
        printf("WGL initialize failed!\n");
        result = -1;
    } else {
        result = 0;
        printf("WGL initialize finished!\n");
    }

    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(m_RenderContext);
    ReleaseDC(TemphWnd, TemphDC);
    DestroyWindow(TemphWnd);

    BaseApplication::Initialize();

    return result;
}

void OpenGLApplication::Finalize() {
    if (m_RenderContext) {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(m_RenderContext);
        m_RenderContext = 0;
    }

    WindowsApplication::Finalize();
}

void OpenGLApplication::Tick() {
    WindowsApplication::Tick();

    // Present the back buffer to the screen since rendering is complete.
    SwapBuffers(m_hDC);
}

void OpenGLApplication::CreateMainWindow() {
    int result;
    auto colorBits =
        m_Config.redBits + m_Config.greenBits +
        m_Config
            .blueBits;  // note on windows this does not include alpha bitplane

    WindowsApplication::CreateMainWindow();

    m_hDC = GetDC(m_hWnd);

    PIXELFORMATDESCRIPTOR m_pfd;

    // now we try to init OpenGL Core profile context
    if (GLAD_WGL_ARB_pixel_format && GLAD_WGL_ARB_multisample &&
        GLAD_WGL_ARB_create_context) {
        // enable MSAA
        const int attributes[] = {WGL_DRAW_TO_WINDOW_ARB,
                                  GL_TRUE,
                                  WGL_SUPPORT_OPENGL_ARB,
                                  GL_TRUE,
                                  WGL_DOUBLE_BUFFER_ARB,
                                  GL_TRUE,
                                  WGL_PIXEL_TYPE_ARB,
                                  WGL_TYPE_RGBA_ARB,
                                  WGL_COLOR_BITS_ARB,
                                  (int)colorBits,
                                  WGL_RED_BITS_ARB,
                                  (int)m_Config.redBits,
                                  WGL_GREEN_BITS_ARB,
                                  (int)m_Config.greenBits,
                                  WGL_BLUE_BITS_ARB,
                                  (int)m_Config.blueBits,
                                  WGL_ALPHA_BITS_ARB,
                                  (int)m_Config.alphaBits,
                                  WGL_DEPTH_BITS_ARB,
                                  (int)m_Config.depthBits,
                                  WGL_STENCIL_BITS_ARB,
                                  (int)m_Config.stencilBits,
                                  WGL_SAMPLE_BUFFERS_ARB,
                                  1,
                                  WGL_SAMPLES_ARB,
                                  m_Config.msaaSamples,
                                  0};

        UINT numFormats;

        if (FAILED(wglChoosePixelFormatARB(m_hDC, attributes, nullptr, 1,
                                           &m_nPixelFormat, &numFormats)) ||
            numFormats == 0) {
            printf("wglChoosePixelFormatARB failed!\n");
        }

        result = SetPixelFormat(m_hDC, m_nPixelFormat, &m_pfd);
        if (result != 1) {
            printf("SetPixelFormat failed!\n");
        }

        const int context_attributes[] = {
            WGL_CONTEXT_MAJOR_VERSION_ARB,
            4,
            WGL_CONTEXT_MINOR_VERSION_ARB,
            3,
            WGL_CONTEXT_FLAGS_ARB,
            WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
            WGL_CONTEXT_PROFILE_MASK_ARB,
            WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
#ifdef OPENGL_RHI_DEBUG
            WGL_CONTEXT_FLAGS_ARB,
            WGL_CONTEXT_DEBUG_BIT_ARB,
#endif
            0};

        m_RenderContext =
            wglCreateContextAttribsARB(m_hDC, 0, context_attributes);
        if (!m_RenderContext) {
            printf("wglCreateContextAttributeARB failed!\n");
        }

        result = wglMakeCurrent(m_hDC, m_RenderContext);
        if (result != 1) {
            printf("wglMakeCurrent failed!\n");
        }

        result = 0;  // we use 0 as success while OpenGL use 1, so convert it
    } else {
        // Set pixel format.
        int m_nPixelFormat = ChoosePixelFormat(m_hDC, &m_pfd);
        if (m_nPixelFormat == 0) {
            printf("ChoosePixelFormat failed!\n");
        }

        result = SetPixelFormat(m_hDC, m_nPixelFormat, &m_pfd);
        if (result != 1) {
            printf("SetPixelFormat failed!\n");
        }

        // Create rendering context.
        m_RenderContext = wglCreateContext(m_hDC);
        if (!m_RenderContext) {
            printf("wglCreateContext failed!\n");
        }

        // Set the rendering context as the current rendering context for this
        // window.
        result = wglMakeCurrent(m_hDC, m_RenderContext);
        if (result != 1) {
            printf("wglMakeCurrent failed!\n");
        }
    }
}
