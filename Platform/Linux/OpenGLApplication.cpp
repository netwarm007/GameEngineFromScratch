#include "OpenGLApplication.hpp"

#include <X11/Xlib-xcb.h>

#include <climits>
#include <cstdio>
#include <cstring>

#include "glad/glad_glx.h"
#include "imgui/examples/imgui_impl_opengl3.h"

using namespace My;
using namespace std;

// Helper to check for extension string presence.  Adapted from:
//   http://www.opengl.org/resources/features/OGLextensions/
static bool isExtensionSupported(const char *extList, const char *extension) {
    const char *start;
    const char *where, *terminator;

    /* Extension names should not have spaces. */
    where = strchr(extension, ' ');
    if (where || *extension == '\0') return false;

    /* It takes a bit of care to be fool-proof about parsing the
       OpenGL extensions string. Don't be fooled by sub-strings,
       etc. */
    for (start = extList;;) {
        where = strstr(start, extension);

        if (!where) break;

        terminator = where + strlen(extension);

        if (where == start || *(where - 1) == ' ')
            if (*terminator == ' ' || *terminator == '\0') return true;

        start = terminator;
    }

    return false;
}

static bool ctxErrorOccurred = false;
static int ctxErrorHandler(Display *dpy, XErrorEvent *ev) {
    ctxErrorOccurred = true;
    return 0;
}

int OpenGLApplication::Initialize() {
    int result;

    /* Open Xlib Display */
    m_pDisplay = XOpenDisplay(NULL);
    if (!m_pDisplay) {
        fprintf(stderr, "Can't open display\n");
    }

    m_nScreen = DefaultScreen(m_pDisplay);

    gladLoadGLX(m_pDisplay, m_nScreen);

    GLXFBConfig *fb_configs;
    int num_fb_configs = 0;

    // Get a matching FB config
    static int visual_attribs[] = {
        GLX_X_RENDERABLE, True, GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_RENDER_TYPE, GLX_RGBA_BIT, GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
        GLX_RED_SIZE, static_cast<int>(INT_MAX & m_Config.redBits),
        GLX_GREEN_SIZE, static_cast<int>(INT_MAX & m_Config.greenBits),
        GLX_BLUE_SIZE, static_cast<int>(INT_MAX & m_Config.blueBits),
        GLX_ALPHA_SIZE, static_cast<int>(INT_MAX & m_Config.alphaBits),
        GLX_DEPTH_SIZE, static_cast<int>(INT_MAX & m_Config.depthBits),
        GLX_STENCIL_SIZE, static_cast<int>(INT_MAX & m_Config.stencilBits),
        GLX_DOUBLEBUFFER, True,
        // GLX_SAMPLE_BUFFERS  , 1,
        // GLX_SAMPLES         , 4,
        None};

    {
        /* Query framebuffer configurations */
        fb_configs = glXChooseFBConfig(m_pDisplay, m_nScreen, visual_attribs,
                                    &num_fb_configs);
        if (!fb_configs || num_fb_configs == 0) {
            fprintf(stderr, "glXGetFBConfigs failed\n");
        }

        /* Pick the FB config/visual with the most samples per pixel */
        {
            int best_fbc = -1, worst_fbc = -1, best_num_samp = -1,
                worst_num_samp = 999;

            for (int i = 0; i < num_fb_configs; ++i) {
                XVisualInfo *vi =
                    glXGetVisualFromFBConfig(m_pDisplay, fb_configs[i]);
                if (vi) {
                    int samp_buf, samples;
                    glXGetFBConfigAttrib(m_pDisplay, fb_configs[i],
                                        GLX_SAMPLE_BUFFERS, &samp_buf);
                    glXGetFBConfigAttrib(m_pDisplay, fb_configs[i], GLX_SAMPLES,
                                        &samples);

                    printf(
                        "  Matching fbconfig %d, visual ID 0x%lx: SAMPLE_BUFFERS = "
                        "%d,"
                        " SAMPLES = %d\n",
                        i, vi->visualid, samp_buf, samples);

                    if (best_fbc < 0 || (samp_buf && samples > best_num_samp))
                        best_fbc = i, best_num_samp = samples;
                    if (worst_fbc < 0 || !samp_buf || samples < worst_num_samp)
                        worst_fbc = i, worst_num_samp = samples;
                }
                XFree(vi);
            }

            fb_config = fb_configs[best_fbc];
        }
    }

    XcbApplication::Initialize(); // implicitly calling create main window here

    /* Get a visual */
    vi = glXGetVisualFromFBConfig(m_pDisplay, fb_config);
    printf("Chosen visual ID = 0x%lx\n", vi->visualid);

    /* establish connection to X server */
    m_pConn = XGetXCBConnection(m_pDisplay);
    if (!m_pConn) {
        XCloseDisplay(m_pDisplay);
        fprintf(stderr, "Can't get xcb connection from display\n");
        return -1;
    }

    /* Acquire event queue ownership */
    XSetEventQueueOwner(m_pDisplay, XCBOwnsEventQueue);

    /* Find XCB screen */
    xcb_screen_iterator_t screen_iter =
        xcb_setup_roots_iterator(xcb_get_setup(m_pConn));
    for (int screen_num = vi->screen; screen_iter.rem && screen_num > 0;
         --screen_num, xcb_screen_next(&screen_iter))
        ;
    m_pScreen = screen_iter.data;
    m_nVi = vi->visualid;

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(m_Config.screenWidth), 
                            static_cast<float>(m_Config.screenHeight)); 

    ImGui::StyleColorsDark();

    return result;
}

void OpenGLApplication::CreateMainWindow() {
    XcbApplication::CreateMainWindow();

    /* Create OpenGL context */
    ctxErrorOccurred = false;
    int (*oldHandler)(Display *, XErrorEvent *) =
        XSetErrorHandler(&ctxErrorHandler);

    /* Get the default screen's GLX extension list */
    const char *glxExts;
    glxExts = glXQueryExtensionsString(m_pDisplay, m_nScreen);

    if (!isExtensionSupported(glxExts, "GLX_ARB_create_context") ||
        !glXCreateContextAttribsARB) {
        printf(
            "glXCreateContextAttribsARB() not found"
            " ... using old-style GLX context\n");
        m_Context =
            glXCreateNewContext(m_pDisplay, fb_config, GLX_RGBA_TYPE, 0, True);
        if (!m_Context) {
            fprintf(stderr, "glXCreateNewContext failed\n");
        }
    } else {
        int context_attribs[] = {GLX_CONTEXT_MAJOR_VERSION_ARB,
                                 4,
                                 GLX_CONTEXT_MINOR_VERSION_ARB,
                                 3,
#ifdef OPENGL_RHI_DEBUG
                                 GLX_CONTEXT_FLAGS_ARB,
                                 GLX_CONTEXT_DEBUG_BIT_ARB,
#endif
                                 None};

        printf("Creating context\n");
        m_Context = glXCreateContextAttribsARB(m_pDisplay, fb_config, 0, True,
                                               context_attribs);

        XSync(m_pDisplay, False);
        if (!ctxErrorOccurred && m_Context)
            printf("Created GL 4.3 context\n");
        else {
            /* GLX_CONTEXT_MAJOR_VERSION_ARB = 1 */
            context_attribs[1] = 1;
            /* GLX_CONTEXT_MINOR_VERSION_ARB = 0 */
            context_attribs[3] = 0;

            ctxErrorOccurred = false;

            printf(
                "Failed to create GL 4.3 context"
                " ... using old-style GLX context\n");
            m_Context = glXCreateContextAttribsARB(m_pDisplay, fb_config, 0,
                                                   True, context_attribs);
        }
    }

    XSync(m_pDisplay, False);

    XSetErrorHandler(oldHandler);

    if (ctxErrorOccurred || !m_Context) {
        printf("Failed to create an OpenGL context\n");
    }

    /* Verifying that context is a direct context */
    if (!glXIsDirect(m_pDisplay, m_Context)) {
        printf("Indirect GLX rendering context obtained\n");
    } else {
        printf("Direct GLX rendering context obtained\n");
    }

    /* Create GLX Window */
    GLXWindow glxwindow = glXCreateWindow(m_pDisplay, fb_config, m_Window, 0);

    if (!glxwindow) {
        xcb_destroy_window(m_pConn, m_Window);
        glXDestroyContext(m_pDisplay, m_Context);

        fprintf(stderr, "glxCreateWindow failed\n");
    }

    m_Drawable = glxwindow;

    /* make OpenGL context current */
    if (!glXMakeContextCurrent(m_pDisplay, m_Drawable, m_Drawable, m_Context)) {
        xcb_destroy_window(m_pConn, m_Window);
        glXDestroyContext(m_pDisplay, m_Context);

        fprintf(stderr, "glXMakeContextCurrent failed\n");
    }

    XFree(vi);
}

void OpenGLApplication::Tick() {
    glXSwapBuffers(m_pDisplay, m_Drawable);
    XcbApplication::Tick();
}

void OpenGLApplication::Finalize() {
    ImGui::DestroyContext();
}
