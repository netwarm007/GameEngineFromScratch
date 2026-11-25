// Windows regression: an RHI switch recreates the host window at the size and
// position it had before the switch (M4.3 preserved-geometry contract that the
// SDL/XCB/Cocoa hosts already implement). No GPU or graphics session needed:
// only CreateMainWindow/FinalizePlatformResources are exercised.
#include "Platform/Windows/WindowsApplication.hpp"

#include <cstdio>

using namespace My;

namespace {

void PumpMessages(DWORD duration_ms) {
    ULONGLONG start = GetTickCount64();
    MSG msg;
    while (GetTickCount64() - start < duration_ms) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        Sleep(10);
    }
}

HWND CurrentWindow(WindowsApplication& app) {
    PumpMessages(50);
    return static_cast<HWND>(app.GetMainWindowHandler());
}

bool RectsEqual(const RECT& a, const RECT& b) {
    return a.left == b.left && a.top == b.top && a.right == b.right &&
           a.bottom == b.bottom;
}

void PrintRect(const char* label, const RECT& r) {
    std::printf("  %s = (%ld, %ld, %ld, %ld)\n", label, r.left, r.top,
                r.right, r.bottom);
}

int Fail(const char* what) {
    std::printf("FAIL: %s\n", what);
    return 1;
}

}  // namespace

int main() {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                            "Window Geometry Preservation Test");
    WindowsApplication app(config);

    // 1) Move/resize the initial window, then simulate the window teardown and
    //    recreation of an RHI switch.
    app.CreateMainWindow();
    HWND first = CurrentWindow(app);
    if (!first) return Fail("initial CreateMainWindow produced no window");

    const RECT expected{140, 120, 140 + 1040, 120 + 720};
    if (!SetWindowPos(first, nullptr, expected.left, expected.top,
                      expected.right - expected.left,
                      expected.bottom - expected.top,
                      SWP_NOZORDER | SWP_NOACTIVATE)) {
        return Fail("SetWindowPos failed");
    }
    PumpMessages(100);

    app.FinalizePlatformResources();
    app.CreateMainWindow();
    HWND second = CurrentWindow(app);
    if (!second) return Fail("recreated window missing");
    if (second == first) return Fail("switch must recreate the native window");

    RECT got{};
    GetWindowRect(second, &got);
    if (!RectsEqual(got, expected)) {
        std::printf("FAIL: recreated geometry not preserved\n");
        PrintRect("expected", expected);
        PrintRect("actual", got);
        app.FinalizePlatformResources();
        return Fail("size/position memory broken");
    }
    std::printf("PASS: size and position survive window recreation\n");

    // 2) A maximized window stays maximized, with the normal placement restored.
    ShowWindow(second, SW_SHOWMAXIMIZED);
    PumpMessages(200);

    app.FinalizePlatformResources();
    app.CreateMainWindow();
    HWND third = CurrentWindow(app);
    if (!third) return Fail("recreated window missing after maximize case");

    WINDOWPLACEMENT placement{};
    placement.length = sizeof(placement);
    GetWindowPlacement(third, &placement);
    if (placement.showCmd != SW_SHOWMAXIMIZED) {
        std::printf("FAIL: maximized state lost across recreation (showCmd=%u)\n",
                    placement.showCmd);
        app.FinalizePlatformResources();
        return 1;
    }
    if (!RectsEqual(placement.rcNormalPosition, expected)) {
        std::printf("FAIL: normal placement not restored for maximized window\n");
        PrintRect("expected", expected);
        PrintRect("actual", placement.rcNormalPosition);
        app.FinalizePlatformResources();
        return 1;
    }
    std::printf("PASS: maximized state and normal placement survive recreation\n");

    // 3) A minimized window returns as normal: keeping it minimized would stall
    //    the switch-complete handshake on a zero-size framebuffer.
    ShowWindow(third, SW_RESTORE);
    PumpMessages(100);
    ShowWindow(third, SW_SHOWMINIMIZED);
    PumpMessages(100);

    app.FinalizePlatformResources();
    app.CreateMainWindow();
    HWND fourth = CurrentWindow(app);
    if (!fourth) return Fail("recreated window missing after minimize case");

    GetWindowPlacement(fourth, &placement);
    if (placement.showCmd != SW_SHOWNORMAL) {
        std::printf("FAIL: minimized window should reappear normal (showCmd=%u)\n",
                    placement.showCmd);
        app.FinalizePlatformResources();
        return 1;
    }
    if (!RectsEqual(placement.rcNormalPosition, expected)) {
        std::printf("FAIL: placement lost through minimize case\n");
        PrintRect("expected", expected);
        PrintRect("actual", placement.rcNormalPosition);
        app.FinalizePlatformResources();
        return 1;
    }
    std::printf("PASS: minimized window restores to preserved placement\n");

    app.FinalizePlatformResources();
    std::printf("All window geometry preservation checks passed.\n");
    return 0;
}
