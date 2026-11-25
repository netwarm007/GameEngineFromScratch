// Linux XCB regression for the RHI-switch preserved-geometry contract (the
// XCB counterpart of WindowGeometryPreservationTest on Windows). Capturing the
// retiring window must yield root (screen) coordinates, not the parent-frame
// coordinates xcb_get_geometry reports under a reparenting window manager,
// and the recreated window must carry ICCCM user-specified position hints so
// the WM restores the placement instead of applying its own policy.
//
// Needs an X display with a reparenting window manager: without one the
// decoration-offset failure mode does not exist, so the test reports SKIP
// (ctest SKIP_RETURN_CODE 77). The dedicated CI step guarantees a window
// manager and sets GEFS_REQUIRE_WM=1, which turns "no WM" into a failure.
#include "Platform/Linux/XcbApplication.hpp"

#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <xcb/xcb.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

using namespace My;

namespace {

constexpr int kSkip = 77;
constexpr int kTolerancePx = 2;

int SilentXError(Display*, XErrorEvent*) { return 0; }

bool WmRunning(Display* dpy) {
    Atom atom = XInternAtom(dpy, "_NET_SUPPORTING_WM_CHECK", False);
    Window root = DefaultRootWindow(dpy);
    Atom actual = None;
    int format = 0;
    unsigned long items = 0, after = 0;
    unsigned char* data = nullptr;
    bool running = false;
    if (XGetWindowProperty(dpy, root, atom, 0, 1, False, XA_WINDOW, &actual,
                           &format, &items, &after, &data) == Success &&
        actual == XA_WINDOW && items > 0) {
        running = *reinterpret_cast<Window*>(data) != None;
    }
    if (data) XFree(data);
    return running;
}

struct WinRect {
    int x = 0, y = 0;
    unsigned width = 0, height = 0;
};

WinRect ClientRootGeometry(Display* dpy, xcb_window_t win) {
    WinRect g;
    int absX = 0, absY = 0;
    Window child = None;
    XWindowAttributes attrs{};
    XSync(dpy, False);
    if (XTranslateCoordinates(dpy, static_cast<Window>(win),
                              DefaultRootWindow(dpy), 0, 0, &absX, &absY,
                              &child) &&
        XGetWindowAttributes(dpy, static_cast<Window>(win), &attrs)) {
        g.x = absX;
        g.y = absY;
        g.width = static_cast<unsigned>(attrs.width);
        g.height = static_cast<unsigned>(attrs.height);
    }
    return g;
}

// A freshly mapped window is reparented and positioned asynchronously by the
// WM; return the placement once two consecutive samples agree (or the last
// one after the deadline so callers still get printable numbers on failure).
WinRect WaitForSettledPlacement(Display* dpy, xcb_window_t win) {
    WinRect prev = ClientRootGeometry(dpy, win);
    for (int i = 0; i < 60; ++i) {
        usleep(50 * 1000);
        WinRect now = ClientRootGeometry(dpy, win);
        if (now.x == prev.x && now.y == prev.y && now.width == prev.width &&
            now.height == prev.height) {
            return now;
        }
        prev = now;
    }
    return prev;
}

bool GeometryClose(const WinRect& a, const WinRect& b, int tol) {
    return abs(a.x - b.x) <= tol && abs(a.y - b.y) <= tol &&
           abs(static_cast<int>(a.width) - static_cast<int>(b.width)) <= tol &&
           abs(static_cast<int>(a.height) - static_cast<int>(b.height)) <= tol;
}

void PrintGeometry(const char* label, const WinRect& g) {
    std::printf("  %s = (%d, %d, %ux%u)\n", label, g.x, g.y, g.width,
                g.height);
}

xcb_window_t CurrentWindow(XcbApplication& app) {
    return static_cast<xcb_window_t>(
        reinterpret_cast<uintptr_t>(app.GetMainWindowHandler()));
}

}  // namespace

int main() {
    if (!getenv("DISPLAY")) {
        std::printf("SKIP: no DISPLAY for the XCB geometry test\n");
        return kSkip;
    }
    XSetErrorHandler(SilentXError);
    Display* dpy = XOpenDisplay(NULL);
    if (!dpy) {
        std::printf("SKIP: cannot open the X display\n");
        return kSkip;
    }
    if (!WmRunning(dpy)) {
        if (getenv("GEFS_REQUIRE_WM")) {
            std::printf("FAIL: GEFS_REQUIRE_WM is set but no WM owns the "
                        "display\n");
            XCloseDisplay(dpy);
            return 1;
        }
        std::printf("SKIP: no window manager on the display (the plain "
                    "xvfb suite has no reparenting failure mode)\n");
        XCloseDisplay(dpy);
        return kSkip;
    }

    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                            "Xcb Window WinRect Preservation Test");
    XcbApplication app(config);

    // 1) Move/resize the initial window through the WM, then simulate the
    //    retire/recreate of an RHI switch. The recreated client must land at
    //    the same root coordinates and size, within a couple of pixels.
    app.CreateMainWindow();
    xcb_window_t first = CurrentWindow(app);
    if (first == XCB_WINDOW_NONE) {
        std::printf("FAIL: initial CreateMainWindow produced no window\n");
        XCloseDisplay(dpy);
        return 1;
    }
    WaitForSettledPlacement(dpy, first);
    XMoveResizeWindow(dpy, static_cast<Window>(first), 140, 120, 1040, 720);
    XFlush(dpy);
    WinRect expected = WaitForSettledPlacement(dpy, first);

    app.FinalizePlatformResources();
    app.CreateMainWindow();
    xcb_window_t second = CurrentWindow(app);
    if (second == XCB_WINDOW_NONE) {
        std::printf("FAIL: recreated window missing\n");
        XCloseDisplay(dpy);
        return 1;
    }
    if (second == first) {
        std::printf("FAIL: switch must recreate the native window\n");
        app.FinalizePlatformResources();
        XCloseDisplay(dpy);
        return 1;
    }
    WinRect got = WaitForSettledPlacement(dpy, second);
    if (!GeometryClose(got, expected, kTolerancePx)) {
        std::printf("FAIL: recreated geometry not preserved\n");
        PrintGeometry("expected", expected);
        PrintGeometry("actual", got);
        app.FinalizePlatformResources();
        XCloseDisplay(dpy);
        return 1;
    }
    std::printf("PASS: size and position survive window recreation\n");

    // 2) The recreated window must publish ICCCM position hints; WMs ignore
    //    create-window coordinates without them.
    Atom prop = XInternAtom(dpy, "WM_NORMAL_HINTS", False);
    Atom type = XInternAtom(dpy, "WM_SIZE_HINTS", False);
    Atom actual = None;
    int format = 0;
    unsigned long items = 0, after = 0;
    unsigned char* data = nullptr;
    int rc = XGetWindowProperty(dpy, static_cast<Window>(second), prop, 0, 18,
                                False, type, &actual, &format, &items, &after,
                                &data);
    bool hintsOk = rc == Success && actual == type && format == 32 &&
                    items >= 18 && data != nullptr &&
                    (reinterpret_cast<long*>(data)[0] & 1 /*USPosition*/) != 0;
    if (hintsOk) {
        long* hints = reinterpret_cast<long*>(data);
        int hx = static_cast<int>(hints[1]), hy = static_cast<int>(hints[2]);
        if (abs(hx - got.x) > kTolerancePx || abs(hy - got.y) > kTolerancePx) {
            hintsOk = false;
            std::printf("  hints (x, y) = (%d, %d) vs actual (%d, %d)\n", hx,
                        hy, got.x, got.y);
        }
    }
    if (data) XFree(data);
    if (!hintsOk) {
        std::printf("FAIL: recreated window lacks usable WM_NORMAL_HINTS "
                    "user-specified position\n");
        app.FinalizePlatformResources();
        XCloseDisplay(dpy);
        return 1;
    }
    std::printf("PASS: recreated window carries ICCCM user-specified "
                "position hints\n");

    app.FinalizePlatformResources();
    XCloseDisplay(dpy);
    std::printf("All Xcb window geometry preservation checks passed.\n");
    return 0;
}
