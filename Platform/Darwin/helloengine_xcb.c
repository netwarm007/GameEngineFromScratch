#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <xcb/xcb.h>

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

	char title[] = "Hello, Engine!";
	char title_icon[] = "Hello, Engine! (iconified)";

	/* establish connection to X server */
	pConn = xcb_connect(0, 0);

	/* get the first screen */
	pScreen = xcb_setup_roots_iterator(xcb_get_setup(pConn)).data;

	/* get the root window */
	window = pScreen->root;

	/* create black (foreground) graphic context */
	foreground = xcb_generate_id(pConn);
	mask = XCB_GC_FOREGROUND | XCB_GC_GRAPHICS_EXPOSURES;
	values[0] = pScreen->black_pixel;
	values[1] = 0;
	xcb_create_gc(pConn, foreground, window, mask, values);

	/* create which (background) graphic context */
	background = xcb_generate_id(pConn);
	mask = XCB_GC_BACKGROUND | XCB_GC_GRAPHICS_EXPOSURES;
	values[0] = pScreen->white_pixel;
	values[1] = 0;
	xcb_create_gc(pConn, background, window, mask, values);

	/* create window */
	window = xcb_generate_id(pConn);
	mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
	values[0] = pScreen->white_pixel;
	values[1] = XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_KEY_PRESS;
	xcb_create_window (pConn,					/* connection */
					   XCB_COPY_FROM_PARENT,	/* depth */
					   window,					/* window ID */
					   pScreen->root,			/* parent window */
					   20, 20,					/* x, y */
					   640, 480,				/* width, height */
					   10,						/* boarder width */
					   XCB_WINDOW_CLASS_INPUT_OUTPUT, /* class */
					   pScreen->root_visual,	/* visual */
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

	while((pEvent = xcb_wait_for_event(pConn)) && !isQuit) {
		switch(pEvent->response_type & ~0x80) {
		case XCB_EXPOSE:
		    {		
			xcb_rectangle_t rect = { 20, 20, 60, 80 };
			xcb_poly_fill_rectangle(pConn, window, foreground, 1, &rect);
			xcb_flush(pConn);
		    }
			break;
		case XCB_KEY_PRESS:
			isQuit = 1;
			break;
		}
		free(pEvent);
	}

	xcb_disconnect(pConn);

	return 0;
}
	
