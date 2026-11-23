#include "Win32OpenGLBackend.hpp"

#include <stdio.h>
#include <tchar.h>

#include "IApplication.hpp"
#include "IPlatformHost.hpp"
#include "glad/glad_wgl.h"
#include "imgui.h"
#include "imgui_impl_win32.h"

using namespace My;

namespace {
struct WglViewportData {
    HWND window;
    HDC dc;
    void (*render)(ImGuiViewport*, void*);
    bool rendered = false;
};
}  // namespace

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

Win32OpenGLBackend::Win32OpenGLBackend(IApplication& application,
                                       IPlatformHost& platform)
    : m_Application(application), m_Platform(platform) {}

void Win32OpenGLBackend::CreateMainWindow() {
    m_Application.CreateMainWindow();
}

void Win32OpenGLBackend::FinalizeGraphicsResources() {
    if (m_RenderContext) {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(m_RenderContext);
        m_RenderContext = 0;
    }
    if (m_hDC) {
        ReleaseDC((HWND)m_Application.GetMainWindowHandler(), m_hDC);
        m_hDC = nullptr;
    }
}

void Win32OpenGLBackend::WaitForGraphicsIdle() {
    // The WGL platform library links only the WGL loader, not the GL loader
    // that defines glFinish, and WGL exposes no wait entry point. Context
    // deletion at teardown keeps the existing synchronization semantics;
    // explicit GL drain belongs to the OpenGL graphics manager, which links
    // the GL loader.
}

int Win32OpenGLBackend::InitializeGuiPlatform() {
    if (!ImGui::GetCurrentContext()) return 0;
    HWND hWnd = (HWND)m_Application.GetMainWindowHandler();
    if (!hWnd || ImGui::GetIO().BackendPlatformUserData) return -1;
    if (!ImGui_ImplWin32_InitForOpenGL(hWnd)) return -1;
    ImGui_ImplWin32_EnableDpiAwareness();
    return 0;
}

void Win32OpenGLBackend::FinalizeGuiPlatform() {
    m_Platform.FinalizeGuiPlatform();
}

void Win32OpenGLBackend::OnSessionActivated() {
    if (!ImGui::GetCurrentContext()) return;

    auto& platform = ImGui::GetPlatformIO();
    auto* mainViewport = ImGui::GetMainViewport();
    if (!platform.Renderer_RenderWindow) return;
    HWND hWnd = (HWND)m_Application.GetMainWindowHandler();
    // After a window-recreating switch the previous WglViewportData still
    // references the destroyed HWND/DC (the ImGui context survives
    // reconstruction). Refresh it so the main viewport renders to the new
    // window instead of presenting a stale drawable (Finding 4: a recovered
    // OpenGL window stayed unpainted at the class background brush).
    if (mainViewport->RendererUserData) {
        delete static_cast<WglViewportData*>(mainViewport->RendererUserData);
        mainViewport->RendererUserData = nullptr;
    }
    mainViewport->RendererUserData =
        new WglViewportData{hWnd, m_hDC, platform.Renderer_RenderWindow};

    platform.Renderer_CreateWindow = [](ImGuiViewport* viewport) {
        auto* main = static_cast<WglViewportData*>(ImGui::GetMainViewport()->RendererUserData);
        auto hwnd = static_cast<HWND>(viewport->PlatformHandleRaw);
        if (!main || !hwnd) {
            fprintf(stderr, "Cannot create WGL drawable for ImGui viewport\n");
            return;
        }
        HDC dc = GetDC(hwnd);
        const int format = GetPixelFormat(main->dc);
        PIXELFORMATDESCRIPTOR descriptor{};
        // WGL can bind the same context to another DC only with a matching pixel format.
        if (!dc || !format ||
            !DescribePixelFormat(dc, format, sizeof(descriptor), &descriptor) ||
            !SetPixelFormat(dc, format, &descriptor)) {
            fprintf(stderr, "Cannot set pixel format for ImGui WGL viewport\n");
            if (dc) ReleaseDC(hwnd, dc);
            return;
        }
        viewport->RendererUserData = new WglViewportData{hwnd, dc, main->render};
    };
    platform.Renderer_DestroyWindow = [](ImGuiViewport* viewport) {
        auto* data = static_cast<WglViewportData*>(viewport->RendererUserData);
        if (!data) return;
        if (viewport != ImGui::GetMainViewport()) ReleaseDC(data->window, data->dc);
        delete data;
        viewport->RendererUserData = nullptr;
    };
    platform.Renderer_RenderWindow = [](ImGuiViewport* viewport, void* arg) {
        auto* data = static_cast<WglViewportData*>(viewport->RendererUserData);
        if (!data) return;
        data->rendered = false;
        const HDC previousDC = wglGetCurrentDC();
        const HGLRC context = wglGetCurrentContext();
        if (!context) {
            fprintf(stderr, "No current WGL context for ImGui viewport\n");
            return;
        }
        if (wglMakeCurrent(data->dc, context)) {
            data->render(viewport, arg);
            data->rendered = true;
        } else {
            fprintf(stderr, "Cannot bind WGL context to ImGui viewport\n");
        }
        if (!wglMakeCurrent(previousDC, context)) {
            fprintf(stderr, "Cannot restore WGL context after drawing ImGui viewport\n");
        }
    };
    platform.Renderer_SwapBuffers = [](ImGuiViewport* viewport, void*) {
        auto* data = static_cast<WglViewportData*>(viewport->RendererUserData);
        if (data && data->rendered && !SwapBuffers(data->dc)) {
            fprintf(stderr, "Cannot present ImGui WGL viewport\n");
        }
    };
}

void Win32OpenGLBackend::PresentFrame() {
    // Present the back buffer to the screen since rendering is complete.
    // Diagnostic (Finding 4): log before SwapBuffers so a vsync stall on an
    // invalid drawable is captured, with the DC pixel format / double-buffer flag.
    static unsigned s_presentCount = 0;
    if (s_presentCount < 3 || (s_presentCount % 120) == 0) {
        const int pf = m_hDC ? GetPixelFormat(m_hDC) : 0;
        PIXELFORMATDESCRIPTOR pfd{};
        const bool db = pf && DescribePixelFormat(m_hDC, pf, sizeof(pfd), &pfd) &&
                        (pfd.dwFlags & PFD_DOUBLEBUFFER);
        fprintf(stderr,
                "[WGL] PresentFrame #%u enter dc=%p hwnd=%p pf=%d doublebuf=%d "
                "curRC=%p curDC=%p\n",
                s_presentCount, (void*)m_hDC,
                (void*)m_Application.GetMainWindowHandler(), pf, (int)db,
                (void*)wglGetCurrentContext(), (void*)wglGetCurrentDC());
    }
    const BOOL ok = SwapBuffers(m_hDC);
    if (s_presentCount < 3 || !ok) {
        fprintf(stderr, "[WGL] PresentFrame #%u SwapBuffers=%d\n", s_presentCount, (int)ok);
    }
    ++s_presentCount;
}

void Win32OpenGLBackend::BeginFrame() {
    // ImGui multi-viewport rendering binds a viewport DC and restores the prior
    // DC, which can leave the main render context not current on m_hDC after a
    // switch/recovery. Re-bind it so scene passes render into m_hDC's back
    // buffer and PresentFrame's SwapBuffers(m_hDC) presents that same buffer;
    // otherwise the back buffer has the scene but the window stays gray.
    if (m_hDC && m_RenderContext) {
        wglMakeCurrent(m_hDC, m_RenderContext);
    }
}

int Win32OpenGLBackend::InitializeGraphicsResources() {
    HWND hWnd = (HWND)m_Application.GetMainWindowHandler();
    if (!hWnd) {
        fprintf(stderr, "OpenGL initialization requires a native window\n");
        return -1;
    }
    const auto config = m_Application.GetConfiguration();
    auto colorBits = config.redBits + config.greenBits + config.blueBits;

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
    m_pfd.cRedBits = config.redBits;
    m_pfd.cGreenBits = config.greenBits;
    m_pfd.cBlueBits = config.blueBits;
    m_pfd.cAlphaBits = config.alphaBits;
    m_pfd.cDepthBits = config.depthBits;
    m_pfd.cStencilBits = config.stencilBits;
    m_pfd.iLayerType = PFD_MAIN_PLANE;

    HDC TemphDC = GetDC(TemphWnd);
    // Set a temporary default pixel format.
    m_nPixelFormat = ChoosePixelFormat(TemphDC, &m_pfd);
    HGLRC tempContext = nullptr;
    if (TemphDC && m_nPixelFormat &&
        SetPixelFormat(TemphDC, m_nPixelFormat, &m_pfd)) {
        tempContext = wglCreateContext(TemphDC);
    }
    const bool tempCurrent = tempContext && wglMakeCurrent(TemphDC, tempContext);
    const bool loaded = tempCurrent && gladLoadWGL(TemphDC);
    if (tempCurrent) wglMakeCurrent(nullptr, nullptr);
    if (tempContext) wglDeleteContext(tempContext);
    if (TemphDC) ReleaseDC(TemphWnd, TemphDC);
    if (TemphWnd) DestroyWindow(TemphWnd);
    if (!loaded) {
        fprintf(stderr, "Cannot initialize WGL bootstrap context\n");
        return -1;
    }

    m_hDC = GetDC(hWnd);

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
                                  (const int)colorBits,
                                  WGL_RED_BITS_ARB,
                                  (const int)config.redBits,
                                  WGL_GREEN_BITS_ARB,
                                  (const int)config.greenBits,
                                  WGL_BLUE_BITS_ARB,
                                  (const int)config.blueBits,
                                  WGL_ALPHA_BITS_ARB,
                                  (const int)config.alphaBits,
                                  WGL_DEPTH_BITS_ARB,
                                  (const int)config.depthBits,
                                  WGL_STENCIL_BITS_ARB,
                                  (const int)config.stencilBits,
                                  WGL_SAMPLE_BUFFERS_ARB,
                                  1,
                                  WGL_SAMPLES_ARB,
                                  (const int)config.msaaSamples,
                                  0};

        UINT numFormats = 0;
        if (!wglChoosePixelFormatARB(m_hDC, attributes, nullptr, 1u,
                                     &m_nPixelFormat, &numFormats) ||
            numFormats == 0) {
            m_nPixelFormat = ChoosePixelFormat(m_hDC, &m_pfd);
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

        if (m_nPixelFormat && SetPixelFormat(m_hDC, m_nPixelFormat, &m_pfd)) {
            m_RenderContext = wglCreateContextAttribsARB(m_hDC, nullptr, context_attributes);
        }
    } else {
        // Set pixel format.
        m_nPixelFormat = ChoosePixelFormat(m_hDC, &m_pfd);
        if (m_nPixelFormat && SetPixelFormat(m_hDC, m_nPixelFormat, &m_pfd)) {
            m_RenderContext = wglCreateContext(m_hDC);
        }
    }

    if (!m_RenderContext || !wglMakeCurrent(m_hDC, m_RenderContext)) {
        fprintf(stderr, "Cannot initialize WGL context for the main window\n");
        FinalizeGraphicsResources();
        return -1;
    }

    if (GLAD_WGL_EXT_swap_control) {
        wglSwapIntervalEXT(1);
    }
    {
        PIXELFORMATDESCRIPTOR pfd{};
        const int pf = GetPixelFormat(m_hDC);
        DescribePixelFormat(m_hDC, pf, sizeof(pfd), &pfd);
        fprintf(stderr,
                "[WGL] InitGraphics hwnd=%p dc=%p rc=%p chosenPF=%d actualPF=%d "
                "flags=0x%x doublebuf=%d swapInterval=%d\n",
                (void*)hWnd, (void*)m_hDC, (void*)m_RenderContext, m_nPixelFormat,
                pf, (unsigned)pfd.dwFlags, (int)((pfd.dwFlags & PFD_DOUBLEBUFFER) != 0),
                GLAD_WGL_EXT_swap_control ? wglGetSwapIntervalEXT() : -1);
    }
    return 0;
}