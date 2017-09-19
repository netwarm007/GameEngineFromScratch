#include <stdio.h>
#include <climits>
#include <cstring>
#include <X11/Xlib-xcb.h>
#include "OpenGLApplication.hpp"
#include "OpenGL/OpenGLGraphicsManager.hpp"
#include "MemoryManager.hpp"

using namespace My;

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 0, 960, 540, "Game Engine From Scratch (Linux)");
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

    int default_screen;
    GLXFBConfig *fb_configs;
    GLXFBConfig fb_config;
    int num_fb_configs = 0;
    XVisualInfo *vi;
    GLXWindow glxwindow;
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

    /* Open Xlib Display */ 
    m_pDisplay = XOpenDisplay(NULL);
    if(!m_pDisplay)
    {
        fprintf(stderr, "Can't open display\n");
        return -1;
    }

    default_screen = DefaultScreen(m_pDisplay);

    gladLoadGLX(m_pDisplay, default_screen);

    /* Query framebuffer configurations */
    fb_configs = glXChooseFBConfig(m_pDisplay, default_screen, visual_attribs, &num_fb_configs);
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
            XVisualInfo *vi = glXGetVisualFromFBConfig(m_pDisplay, fb_configs[i]);
            if (vi)
            {
                int samp_buf, samples;
                glXGetFBConfigAttrib(m_pDisplay, fb_configs[i], GLX_SAMPLE_BUFFERS, &samp_buf);
                glXGetFBConfigAttrib(m_pDisplay, fb_configs[i], GLX_SAMPLES, &samples);
      
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
    vi = glXGetVisualFromFBConfig(m_pDisplay, fb_config);
    printf("Chosen visual ID = 0x%lx\n", vi->visualid);

    /* establish connection to X server */
    m_pConn = XGetXCBConnection(m_pDisplay);
    if(!m_pConn)
    {
        XCloseDisplay(m_pDisplay);
        fprintf(stderr, "Can't get xcb connection from display\n");
        return -1;
    }

    /* Acquire event queue ownership */
    XSetEventQueueOwner(m_pDisplay, XCBOwnsEventQueue);

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
    glxExts = glXQueryExtensionsString(m_pDisplay, default_screen);

    /* Create OpenGL context */
    ctxErrorOccurred = false;
    int (*oldHandler)(Display*, XErrorEvent*) =
        XSetErrorHandler(&ctxErrorHandler);

    if (!isExtensionSupported(glxExts, "GLX_ARB_create_context") ||
       !glXCreateContextAttribsARB )
    {
        printf( "glXCreateContextAttribsARB() not found"
            " ... using old-style GLX context\n" );
        m_Context = glXCreateNewContext(m_pDisplay, fb_config, GLX_RGBA_TYPE, 0, True);
        if(!m_Context)
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
        m_Context = glXCreateContextAttribsARB(m_pDisplay, fb_config, 0,
                                          True, context_attribs );

        XSync(m_pDisplay, False);
        if (!ctxErrorOccurred && m_Context)
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
          m_Context = glXCreateContextAttribsARB(m_pDisplay, fb_config, 0, 
                                            True, context_attribs );
        }
    }

    XSync(m_pDisplay, False);

    XSetErrorHandler(oldHandler);

    if (ctxErrorOccurred || !m_Context)
    {
        printf( "Failed to create an OpenGL context\n" );
        return -1;
    }

    /* Verifying that context is a direct context */
    if (!glXIsDirect (m_pDisplay, m_Context))
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
                m_pDisplay,
                fb_config,
                m_Window,
                0
                );

    if(!glxwindow)
    {
        xcb_destroy_window(m_pConn, m_Window);
        glXDestroyContext(m_pDisplay, m_Context);

        fprintf(stderr, "glxCreateWindow failed\n");
        return -1;
    }

    m_Drawable = glxwindow;

    /* make OpenGL context current */
    if(!glXMakeContextCurrent(m_pDisplay, m_Drawable, m_Drawable, m_Context))
    {
        xcb_destroy_window(m_pConn, m_Window);
        glXDestroyContext(m_pDisplay, m_Context);

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

void My::OpenGLApplication::OnDraw()
{
    glXSwapBuffers(m_pDisplay, m_Drawable);
}

