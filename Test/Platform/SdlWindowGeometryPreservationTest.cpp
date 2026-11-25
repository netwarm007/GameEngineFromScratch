// Linux SDL-host counterpart of the RHI-switch preserved-geometry contract:
// the recreated window keeps the user's placement and, in particular, the
// maximized state (Windows host parity; minimized counts as normal because a
// recreated minimized window never yields a non-zero framebuffer). The size
// and position capture/restore already existed; the state round-trip did not.
//
// Needs an X display with a window manager (SDL_MaximizeWindow and the
// decorated position semantics are WM conversations). Without one the test
// reports SKIP (ctest SKIP_RETURN_CODE 77); the dedicated CI step sets
// GEFS_REQUIRE_WM=1, turning "no WM" into a failure.
#include "Platform/Sdl/SdlApplication.hpp"

#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <SDL2/SDL.h>

#include <cstdio>
#include <cstdlib>

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

void PumpEvents(unsigned duration_ms) {
    SDL_Event e;
    const Uint32 start = SDL_GetTicks();
    while (SDL_GetTicks() - start < duration_ms) {
        SDL_PumpEvents();
        while (SDL_PollEvent(&e)) {
        }
        SDL_Delay(10);
    }
}

struct Rect {
    int x = 0, y = 0, w = 0, h = 0;
};

Rect CurrentRect(SDL_Window* win) {
    Rect r;
    SDL_GetWindowPosition(win, &r.x, &r.y);
    SDL_GetWindowSize(win, &r.w, &r.h);
    return r;
}

bool RectClose(const Rect& a, const Rect& b, int tol) {
    return abs(a.x - b.x) <= tol && abs(a.y - b.y) <= tol &&
           abs(a.w - b.w) <= tol && abs(a.h - b.h) <= tol;
}

void PrintRect(const char* label, const Rect& r) {
    std::printf("  %s = (%d, %d, %dx%d)\n", label, r.x, r.y, r.w, r.h);
}

}  // namespace

int main() {
    if (!getenv("DISPLAY")) {
        std::printf("SKIP: no DISPLAY for the SDL geometry test\n");
        return kSkip;
    }
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::printf("SKIP: SDL video init failed (%s)\n", SDL_GetError());
        return kSkip;
    }
    XSetErrorHandler(SilentXError);
    Display* dpy = XOpenDisplay(NULL);
    if (!dpy) {
        std::printf("SKIP: cannot open the X display\n");
        SDL_Quit();
        return kSkip;
    }
    if (!WmRunning(dpy)) {
        XCloseDisplay(dpy);
        if (getenv("GEFS_REQUIRE_WM")) {
            std::printf("FAIL: GEFS_REQUIRE_WM is set but no WM owns the "
                        "display\n");
            SDL_Quit();
            return 1;
        }
        std::printf("SKIP: no window manager on the display\n");
        SDL_Quit();
        return kSkip;
    }
    XCloseDisplay(dpy);

    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                            "Sdl Window Geometry Preservation Test");
    SdlApplication app(config);

    // 1) Move/resize, retire, recreate: the placement must survive.
    app.CreateMainWindow();
    SDL_Window* first = app.GetSdlWindow();
    if (!first) {
        std::printf("FAIL: initial CreateMainWindow produced no window\n");
        SDL_Quit();
        return 1;
    }
    PumpEvents(200);
    SDL_SetWindowPosition(first, 140, 120);
    SDL_SetWindowSize(first, 1040, 720);
    PumpEvents(200);
    const Rect expected = CurrentRect(first);

    app.FinalizePlatformResources();
    app.CreateMainWindow();
    SDL_Window* second = app.GetSdlWindow();
    if (!second) {
        std::printf("FAIL: recreated window missing\n");
        SDL_Quit();
        return 1;
    }
    PumpEvents(200);
    const Rect placed = CurrentRect(second);
    if (!RectClose(placed, expected, kTolerancePx)) {
        std::printf("FAIL: recreated geometry not preserved\n");
        PrintRect("expected", expected);
        PrintRect("actual", placed);
        app.FinalizePlatformResources();
        SDL_Quit();
        return 1;
    }
    std::printf("PASS: size and position survive window recreation\n");

    // 2) A maximized window stays maximized across recreation.
    SDL_MaximizeWindow(second);
    bool maximized = false;
    for (int i = 0; i < 60 && !maximized; ++i) {
        PumpEvents(50);
        maximized = (SDL_GetWindowFlags(second) & SDL_WINDOW_MAXIMIZED) != 0;
    }
    if (!maximized) {
        std::printf("FAIL: WM did not maximize the window (environment)\n");
        app.FinalizePlatformResources();
        SDL_Quit();
        return 1;
    }

    app.FinalizePlatformResources();
    app.CreateMainWindow();
    SDL_Window* third = app.GetSdlWindow();
    if (!third) {
        std::printf("FAIL: recreated window missing after maximize case\n");
        SDL_Quit();
        return 1;
    }
    bool restoredMaximized = false;
    for (int i = 0; i < 60 && !restoredMaximized; ++i) {
        PumpEvents(50);
        restoredMaximized =
            (SDL_GetWindowFlags(third) & SDL_WINDOW_MAXIMIZED) != 0;
    }
    if (!restoredMaximized) {
        std::printf("FAIL: maximized state lost across recreation\n");
        app.FinalizePlatformResources();
        SDL_Quit();
        return 1;
    }
    std::printf("PASS: maximized state survives recreation\n");

    // 3) A minimized window returns as normal so the switch-complete
    //    handshake cannot stall on a zero-size framebuffer.
    SDL_RestoreWindow(third);
    PumpEvents(150);
    const Rect beforeMinimize = CurrentRect(third);
    SDL_MinimizeWindow(third);
    bool minimized = false;
    for (int i = 0; i < 40 && !minimized; ++i) {
        PumpEvents(50);
        minimized = (SDL_GetWindowFlags(third) & SDL_WINDOW_MINIMIZED) != 0;
    }

    app.FinalizePlatformResources();
    app.CreateMainWindow();
    SDL_Window* fourth = app.GetSdlWindow();
    if (!fourth) {
        std::printf("FAIL: recreated window missing after minimize case\n");
        SDL_Quit();
        return 1;
    }
    PumpEvents(200);
    const Uint32 flags = SDL_GetWindowFlags(fourth);
    if (flags & SDL_WINDOW_MINIMIZED) {
        std::printf("FAIL: minimized window should reappear normal\n");
        app.FinalizePlatformResources();
        SDL_Quit();
        return 1;
    }
    const Rect afterMinimizeCase = CurrentRect(fourth);
    if (!RectClose(afterMinimizeCase, beforeMinimize, kTolerancePx)) {
        std::printf("FAIL: placement lost through minimize case\n");
        PrintRect("expected", beforeMinimize);
        PrintRect("actual", afterMinimizeCase);
        app.FinalizePlatformResources();
        SDL_Quit();
        return 1;
    }
    std::printf("PASS: minimized window restores to preserved placement\n");

    app.FinalizePlatformResources();
    SDL_Quit();
    std::printf("All Sdl window geometry preservation checks passed.\n");
    return 0;
}
