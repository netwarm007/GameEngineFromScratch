# K-6 — Window-geometry edge-case matrix (R-1 residual; proposal for user tier confirmation already given)

Derived from the #89 fix experience. Grouped by feasibility lane:
**L1 = CI-automatable (xvfb/openbox on Linux, desktop session on Windows)**,
**L2 = milestone real-desktop spot-check (C-2 track; multi-monitor/DPI/WM-matrix)**.

## L1 (CI matrix, extend existing Xcb/Sdl/Win32 preservation tests)
- E1 negative-coordinate retention: move window to (-600,-400) region via WM-offscreen drag emulation (X: XMoveWindow negative; Win: SetWindowPos negative) -> retire/recreate keeps negative origin (no clamp-to-visible). XCB note: m_PreservedX/Y are int16_t by X protocol; verify -32768..32767 wrap boundaries.
- E2 partially-offscreen window (title strip only visible): normal rect preserved exactly; no auto "snap back" introduced by hints (USPosition honored).
- E3 tiny/minimum-size window: resize to platform minimum (WM floor), retire/recreate lands at the *actual settled* size (our capture reads post-WM values — regression guard for pre-floor values).
- E4 rapid double switch (A→B→A within seconds): two retires/recreates in one app lifetime; geometry memory chain must be transitive (recapture on every retire, no stale cache reuse) — covers LinuxHost dual-mode host specifically.
- E5 switch while window unmapped/iconified mid-flow (Win32 minimized case done in #89; XCB/SDL iconified variants added to CI).
- E6 maximized-at-retire on Windows with *non-primary* work area (CI VM single-monitor approximates: maximize+restore-normal-rect assertions already exist; extend to restore-then-switch order).

## L2 (real desktop, milestone spot-check via 小窗/小台)
- M1 multi-monitor: window on secondary monitor, negative or >primary-width coordinates; RHI switch keeps monitor affinity (DWM/WM policy vs our hints).
- M2 HiDPI/fractional scaling: physical vs logical pixel mismatch (SDL returns logical on some backends; XCB root coords are physical). Baseline: run switch under 150% scaling, assert visual position parity within one title-bar height.
- M3 WM behavior matrix: KDE/KWin + GNOME/Mutter + tiling (i3/sway). Tiling WMs intentionally ignore USPosition — expected outcome defined as "new window managed by tiling policy, no crash/no offscreen orphan"; document as accepted non-preservation.
- M4 display topology change mid-session (unplug monitor while minimized, then switch) — recovery re-places onscreen.
- M5 Wayland/XWayland note: our XCB host under XWayland reports fake-root coordinates; out-of-scope for preservation, but should fail-soft — include observation only.

## Acceptance mapping (R-1 gate)
- L1: extended tests in Test/Platform/*, `Window geometry preservation under WM` step + windows_regression full scope green on a K-6 revision.
- L2: milestone checklist run once on TIM-UMPC (M1/M2/M3) and once on 小台 (M1/M4) when back online; evidence via screenshots + qianwen-vision gate (K-5b tool reuse) where pixels matter.
- Deliberate non-goals: Wayland-native host; fullscreen state memory.

Author: 小企鹅 01750738, 2026-11-26. Awaiting user/PM sequencing vs K-1 load.
