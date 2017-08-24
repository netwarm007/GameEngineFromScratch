#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <X11/Xlib.h>
#include <X11/Xlib-xcb.h>
#include <xcb/xcb.h>

#include <GL/gl.h> 
#include <GL/glx.h> 
#include <GL/glu.h>

void DrawAQuad() {
	glClearColor(1.0, 1.0, 1.0, 1.0); 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	glMatrixMode(GL_PROJECTION); 
	glLoadIdentity(); 
	glOrtho(-1., 1., -1., 1., 1., 20.); 

	glMatrixMode(GL_MODELVIEW); 
	glLoadIdentity(); 
	gluLookAt(0., 0., 10., 0., 0., 0., 0., 1., 0.); 

	glBegin(GL_QUADS); 
	glColor3f(1., 0., 0.); 
	glVertex3f(-.75, -.75, 0.); 
	glColor3f(0., 1., 0.); 
	glVertex3f( .75, -.75, 0.); 
	glColor3f(0., 0., 1.); 
	glVertex3f( .75, .75, 0.); 
	glColor3f(1., 1., 0.); 
	glVertex3f(-.75, .75, 0.); 
	glEnd(); 
} 

int main(void) {
	xcb_connection_t	*pConn;
	xcb_screen_t		*pScreen;
	xcb_window_t		window;
	xcb_gcontext_t		foreground;
	xcb_gcontext_t		background;
	xcb_generic_event_t	*pEvent;
	uint32_t		mask = 0;
	uint32_t		values[2];
	uint8_t			isQuit = 0;

	char title[] = "Hello, Engine![OpenGL]";
	char title_icon[] = "Hello, Engine! (iconified)";

	int visualID = 0;
	GLXContext context;

	Display *display;
        int default_screen;
	xcb_colormap_t colormap;
        GLXFBConfig *fb_configs = 0;
        GLXFBConfig fb_config;
        int num_fb_configs = 0;
        GLXDrawable drawable = 0;
        GLXWindow glxwindow;

        /* Open Xlib Display */ 
        display = XOpenDisplay(0);
        if(!display)
        {
            fprintf(stderr, "Can't open display\n");
            return -1;
        }

        default_screen = DefaultScreen(display);
	/* establish connection to X server */
	pConn = XGetXCBConnection(display);
        if(!pConn)
        {
            XCloseDisplay(display);
            fprintf(stderr, "Can't get xcb connection from display\n");
            return -1;
        }

        /* Acquire event queue ownership */
        XSetEventQueueOwner(display, XCBOwnsEventQueue);

        /* Find XCB screen */
        xcb_screen_iterator_t screen_iter = 
            xcb_setup_roots_iterator(xcb_get_setup(pConn));
        for(int screen_num = default_screen;
            screen_iter.rem && screen_num > 0;
            --screen_num, xcb_screen_next(&screen_iter));
        pScreen = screen_iter.data;

	/* get the root window */
	window = pScreen->root;

        /* Query framebuffer configurations */
        fb_configs = glXGetFBConfigs(display, default_screen, &num_fb_configs);
        if(!fb_configs || num_fb_configs == 0)
        {
            fprintf(stderr, "glXGetFBConfigs failed\n");
            return -1;
        }

        /* Select first framebuffer config and query visualID */
        fb_config = fb_configs[0];
        glXGetFBConfigAttrib(display, fb_config, GLX_VISUAL_ID , &visualID);

        /* Create OpenGL context */
        context = glXCreateNewContext(display, fb_config, GLX_RGBA_TYPE, 0, True);
        if(!context)
        {
            fprintf(stderr, "glXCreateNewContext failed\n");
            return -1;
        }

        /* Create XID's for colormap */
        colormap = xcb_generate_id(pConn);

        /* Create colormap */
        xcb_create_colormap(
            pConn,
            XCB_COLORMAP_ALLOC_NONE,
            colormap,
            window,
            visualID
            );

	/* create window */
	window = xcb_generate_id(pConn);
	mask = XCB_CW_EVENT_MASK  | XCB_CW_COLORMAP;
	values[0] = XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_KEY_PRESS;
	values[1] = colormap;
	xcb_create_window (pConn,					/* connection */
					   XCB_COPY_FROM_PARENT,	/* depth */
					   window,					/* window ID */
					   pScreen->root,			/* parent window */
					   20, 20,					/* x, y */
					   640, 480,				/* width, height */
					   10,						/* boarder width */
					   XCB_WINDOW_CLASS_INPUT_OUTPUT, /* class */
					   visualID,	/* visual */
					   mask, values);			/* masks */

	/* set the title of the window */
	xcb_change_property(pConn, XCB_PROP_MODE_REPLACE, window,
			    XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
			    strlen(title), title);

	/* set the title of the window icon */
	xcb_change_property(pConn, XCB_PROP_MODE_REPLACE, window,
			    XCB_ATOM_WM_ICON_NAME, XCB_ATOM_STRING, 8,
			    strlen(title_icon), title_icon);

	/* map the window on the screen */
	xcb_map_window(pConn, window);

	xcb_flush(pConn);

        /* Create GLX Window */
        glxwindow = 
            glXCreateWindow(
                display,
                fb_config,
                window,
                0
                );

        if(!window)
        {
            xcb_destroy_window(pConn, window);
            glXDestroyContext(display, context);

            fprintf(stderr, "glXDestroyContext failed\n");
            return -1;
        }

        drawable = glxwindow;

        /* make OpenGL context current */
        if(!glXMakeContextCurrent(display, drawable, drawable, context))
        {
            xcb_destroy_window(pConn, window);
            glXDestroyContext(display, context);

            fprintf(stderr, "glXMakeContextCurrent failed\n");
            return -1;
        }


	while((pEvent = xcb_wait_for_event(pConn)) && !isQuit) {
		switch(pEvent->response_type & ~0x80) {
		case XCB_EXPOSE:
		    {		
			DrawAQuad();
                    	glXSwapBuffers(display, drawable);
		    }
			break;
		case XCB_KEY_PRESS:
			isQuit = 1;
			break;
		}
		free(pEvent);
	}


        /* Cleanup */
        glXDestroyWindow(display, glxwindow);

        xcb_destroy_window(pConn, window);

        glXDestroyContext(display, context);

	xcb_disconnect(pConn);
	
        /* Cleanup */
        XCloseDisplay(display);

	return 0;
}
	
