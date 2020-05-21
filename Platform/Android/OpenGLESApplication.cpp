#include "OpenGLESApplication.hpp"

using namespace My;
using namespace std;

int OpenGLESApplication::Initialize() {
    // initialize OpenGL ES and EGL

    const EGLint attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_BLUE_SIZE,    static_cast<EGLint>(m_Config.blueBits),
        EGL_GREEN_SIZE,   static_cast<EGLint>(m_Config.greenBits),
        EGL_RED_SIZE,     static_cast<EGLint>(m_Config.redBits),
        EGL_NONE};
    EGLint w, h, format;
    EGLint numConfigs;
    EGLConfig config;
    EGLSurface surface;
    EGLContext context;

    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    eglInitialize(display, 0, 0);

    eglBindAPI(EGL_OPENGL_ES_API);

    /* Here, the application chooses the configuration it desires.
     * find the best match if possible, otherwise use the very first one
     */
    eglChooseConfig(display, attribs, nullptr, 0, &numConfigs);
    std::unique_ptr<EGLConfig[]> supportedConfigs(new EGLConfig[numConfigs]);
    assert(supportedConfigs);
    eglChooseConfig(display, attribs, supportedConfigs.get(), numConfigs,
                    &numConfigs);
    assert(numConfigs);
    auto i = 0;
    for (; i < numConfigs; i++) {
        auto& cfg = supportedConfigs[i];
        EGLint r, g, b, d;
        if (eglGetConfigAttrib(display, cfg, EGL_RED_SIZE, &r) &&
            eglGetConfigAttrib(display, cfg, EGL_GREEN_SIZE, &g) &&
            eglGetConfigAttrib(display, cfg, EGL_BLUE_SIZE, &b) &&
            eglGetConfigAttrib(display, cfg, EGL_DEPTH_SIZE, &d) &&
            r == static_cast<EGLint>(m_Config.redBits) &&
            g == static_cast<EGLint>(m_Config.greenBits) &&
            b == static_cast<EGLint>(m_Config.blueBits) &&
            d == static_cast<EGLint>(m_Config.depthBits)) {
            config = supportedConfigs[i];
            break;
        }
    }
    if (i == numConfigs) {
        config = supportedConfigs[0];
    }

    /* EGL_NATIVE_VISUAL_ID is an attribute of the EGLConfig that is
     * guaranteed to be accepted by ANativeWindow_setBuffersGeometry().
     * As soon as we picked a EGLConfig, we can safely reconfigure the
     * ANativeWindow buffers to match, using EGL_NATIVE_VISUAL_ID. */
    eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);
    surface = eglCreateWindowSurface(display, config, m_pApp->window, NULL);
    EGLint contextAttributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    context = eglCreateContext(display, config, NULL, contextAttributes);

    if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
        LOGW("Unable to eglMakeCurrent");
        return -1;
    }

    eglQuerySurface(display, surface, EGL_WIDTH, &w);
    eglQuerySurface(display, surface, EGL_HEIGHT, &h);

    cerr << "Screen Width: " << w;
    cerr << "Screen Height: " << h;

    m_Display = display;
    m_Context = context;
    m_Surface = surface;
    m_Width = w;
    m_Height = h;
    m_Config.screenWidth = w;
    m_Config.screenHeight = h;
    m_State.angle = 0;

    return AndroidApplication::Initialize();
}

void OpenGLESApplication::Finalize() {
    if (m_Display != EGL_NO_DISPLAY) {
        eglMakeCurrent(m_Display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                       EGL_NO_CONTEXT);
        if (m_Context != EGL_NO_CONTEXT) {
            eglDestroyContext(m_Display, m_Context);
        }
        if (m_Surface != EGL_NO_SURFACE) {
            eglDestroySurface(m_Display, m_Surface);
        }
        eglTerminate(m_Display);
    }
    m_bAnimating = false;
    m_Display = EGL_NO_DISPLAY;
    m_Context = EGL_NO_CONTEXT;
    m_Surface = EGL_NO_SURFACE;

    AndroidApplication::Finalize();
}

void OpenGLESApplication::Tick() {
    AndroidApplication::Tick();
    eglSwapBuffers(m_Display, m_Surface);
}
