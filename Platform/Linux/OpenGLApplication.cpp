#include <stdio.h>
#include <climits>
#include <cstring>
#include <X11/Xlib-xcb.h>
#include "OpenGLApplication.hpp"
#include "OpenGL/OpenGLGraphicsManager.hpp"
#include "MemoryManager.hpp"
#include "glad/glad_glx.h"

using namespace My;

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 32, 0, 0, 960, 540, "Game Engine From Scratch (Linux)");
    IApplication* g_pApp                = static_cast<IApplication*>(new OpenGLApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new OpenGLGraphicsManager);
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);

}

// Helper to check for extension string presence.  Adapted from:
//   http://www.opengl.org/resources/features/OGLextensions/
static bool isExtensionSupported(const char *extList, const char *extension)
{
  const char *start;
  const char *where, *terminator;
  
  /* Extension names should not have spaces. */
  where = strchr(extension, ' ');
  if (where || *extension == '\0')
    return false;

  /* It takes a bit of care to be fool-proof about parsing the
     OpenGL extensions string. Don't be fooled by sub-strings,
     etc. */
  for (start=extList;;) {
    where = strstr(start, extension);

    if (!where)
      break;

    terminator = where + strlen(extension);

    if ( where == start || *(where - 1) == ' ' )
      if ( *terminator == ' ' || *terminator == '\0' )
        return true;

    start = terminator;
  }

  return false;
}

static bool ctxErrorOccurred = false;
static int ctxErrorHandler(Display *dpy, XErrorEvent *ev)
{
    ctxErrorOccurred = true;
    return 0;
}

int My::OpenGLApplication::Initialize()
{
    int result;

    Display *display;
    int default_screen;
    GLXFBConfig *fb_configs;
    GLXFBConfig fb_config;
    int num_fb_configs = 0;
    XVisualInfo *vi;
    GLXWindow glxwindow;
    GLXContext context;
    GLXDrawable drawable;
    const char *glxExts;

    // Get a matching FB config
    static int visual_attribs[] =
    {
      GLX_X_RENDERABLE    , True,
      GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
      GLX_RENDER_TYPE     , GLX_RGBA_BIT,
      GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
      GLX_RED_SIZE        , static_cast<int>(INT_MAX & m_Config.redBits),
      GLX_GREEN_SIZE      , static_cast<int>(INT_MAX & m_Config.greenBits),
      GLX_BLUE_SIZE       , static_cast<int>(INT_MAX & m_Config.blueBits),
      GLX_ALPHA_SIZE      , static_cast<int>(INT_MAX & m_Config.alphaBits),
      GLX_DEPTH_SIZE      , static_cast<int>(INT_MAX & m_Config.depthBits),
      GLX_STENCIL_SIZE    , static_cast<int>(INT_MAX & m_Config.stencilBits),
      GLX_DOUBLEBUFFER    , True,
      //GLX_SAMPLE_BUFFERS  , 1,
      //GLX_SAMPLES         , 4,
      None
    };

    int glx_major, glx_minor;

    /* Open Xlib Display */ 
    display = XOpenDisplay(NULL);
    if(!display)
    {
        fprintf(stderr, "Can't open display\n");
        return -1;
    }

    default_screen = DefaultScreen(display);

    gladLoadGLX(display, default_screen);

    // FBConfigs were added in GLX version 1.3.
    if (!glXQueryVersion(display, &glx_major, &glx_minor) || 
       ((glx_major == 1) && (glx_minor < 3)) || (glx_major < 1))
    {
        fprintf(stderr, "Invalid GLX version\n");
        return -1;
    }

    /* Query framebuffer configurations */
    fb_configs = glXChooseFBConfig(display, default_screen, visual_attribs, &num_fb_configs);
    if(!fb_configs || num_fb_configs == 0)
    {
        fprintf(stderr, "glXGetFBConfigs failed\n");
        return -1;
    }

    /* Pick the FB config/visual with the most samples per pixel */
    {
        int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;

        for (int i=0; i<num_fb_configs; ++i)
        {
            XVisualInfo *vi = glXGetVisualFromFBConfig(display, fb_configs[i]);
            if (vi)
            {
                int samp_buf, samples;
                glXGetFBConfigAttrib(display, fb_configs[i], GLX_SAMPLE_BUFFERS, &samp_buf);
                glXGetFBConfigAttrib(display, fb_configs[i], GLX_SAMPLES, &samples);
      
                printf( "  Matching fbconfig %d, visual ID 0x%lx: SAMPLE_BUFFERS = %d,"
                        " SAMPLES = %d\n", 
                        i, vi -> visualid, samp_buf, samples);

                if (best_fbc < 0 || (samp_buf && samples > best_num_samp))
                    best_fbc = i, best_num_samp = samples;
                if (worst_fbc < 0 || !samp_buf || samples < worst_num_samp)
                    worst_fbc = i, worst_num_samp = samples;
            }
            XFree( vi );
        }

        fb_config = fb_configs[best_fbc];
    }

    /* Get a visual */
    vi = glXGetVisualFromFBConfig(display, fb_config);
    printf("Chosen visual ID = 0x%lx\n", vi->visualid);

    /* establish connection to X server */
    m_pConn = XGetXCBConnection(display);
    if(!m_pConn)
    {
        XCloseDisplay(display);
        fprintf(stderr, "Can't get xcb connection from display\n");
        return -1;
    }

    /* Acquire event queue ownership */
    XSetEventQueueOwner(display, XCBOwnsEventQueue);

    /* Find XCB screen */
    xcb_screen_iterator_t screen_iter = 
        xcb_setup_roots_iterator(xcb_get_setup(m_pConn));
    for(int screen_num = vi->screen;
        screen_iter.rem && screen_num > 0;
        --screen_num, xcb_screen_next(&screen_iter));
    m_pScreen = screen_iter.data;
    m_nVi = vi->visualid;

    result = XcbApplication::Initialize();
    if (result) {
        printf("Xcb Application initialize failed!");
	return -1;
    }

    /* Get the default screen's GLX extension list */
    glxExts = glXQueryExtensionsString(display, default_screen);

    /* Create OpenGL context */
    ctxErrorOccurred = false;
    int (*oldHandler)(Display*, XErrorEvent*) =
        XSetErrorHandler(&ctxErrorHandler);

    if (!isExtensionSupported(glxExts, "GLX_ARB_create_context") ||
       !glXCreateContextAttribsARB )
    {
        printf( "glXCreateContextAttribsARB() not found"
            " ... using old-style GLX context\n" );
        context = glXCreateNewContext(display, fb_config, GLX_RGBA_TYPE, 0, True);
        if(!context)
        {
            fprintf(stderr, "glXCreateNewContext failed\n");
            return -1;
        }
    }
    else
    {
        int context_attribs[] =
          {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
            GLX_CONTEXT_MINOR_VERSION_ARB, 0,
            None
          };

        printf( "Creating context\n" );
        context = glXCreateContextAttribsARB(display, fb_config, 0,
                                          True, context_attribs );

        XSync(display, False);
        if (!ctxErrorOccurred && context)
          printf( "Created GL 3.0 context\n" );
        else
        {
          /* GLX_CONTEXT_MAJOR_VERSION_ARB = 1 */
          context_attribs[1] = 1;
          /* GLX_CONTEXT_MINOR_VERSION_ARB = 0 */
          context_attribs[3] = 0;

          ctxErrorOccurred = false;

          printf( "Failed to create GL 3.0 context"
                  " ... using old-style GLX context\n" );
          context = glXCreateContextAttribsARB(display, fb_config, 0, 
                                            True, context_attribs );
        }
    }

    XSync(display, False);

    XSetErrorHandler(oldHandler);

    if (ctxErrorOccurred || !context)
    {
        printf( "Failed to create an OpenGL context\n" );
        return -1;
    }

    /* Verifying that context is a direct context */
    if (!glXIsDirect (display, context))
    {
        printf( "Indirect GLX rendering context obtained\n" );
    }
    else
    {
        printf( "Direct GLX rendering context obtained\n" );
    }

    /* Create GLX Window */
    glxwindow = 
            glXCreateWindow(
                display,
                fb_config,
                m_Window,
                0
                );

    if(!m_Window)
    {
        xcb_destroy_window(m_pConn, m_Window);
        glXDestroyContext(display, context);

        fprintf(stderr, "glXDestroyContext failed\n");
        return -1;
    }

    drawable = glxwindow;

    /* make OpenGL context current */
    if(!glXMakeContextCurrent(display, drawable, drawable, context))
    {
        xcb_destroy_window(m_pConn, m_Window);
        glXDestroyContext(display, context);

        fprintf(stderr, "glXMakeContextCurrent failed\n");
        return -1;
    }

    XFree(vi);
    return result;
}

void My::OpenGLApplication::Finalize()
{
    XcbApplication::Finalize();
}

void My::OpenGLApplication::Tick()
{
    XcbApplication::Tick();
}

