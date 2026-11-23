# Engineering lessons

Curated from the runtime-RHI effort (M0–M4) and its CI iteration. Two
categories, per the maintainer's instruction (2026-11-23):

- **Long-term principles** do not depend on the current specification of a
  particular system or tool. They stay valid across upgrades.
- **Current practical notes** are tied to a specific system, image, or tool
  version. Each one names what it depends on and when to re-verify. When that
  dependency changes, re-measure or re-test the note instead of carrying it
  forward unchanged — a stale practical note silently diverges from reality.

Raw session logs live outside the repository; this file is the curated,
reviewable distillation.

## Part 1 — Long-term principles

### C++ language and linkage

- Adding an interface base whose pure virtuals have the same signatures as
  methods already implemented through a *sibling* base does not make the class
  concrete: the overrider must be declared in a class derived from that
  interface. Write explicit delegating overrides.
- A function definition at global scope under `using namespace My;` defines a
  *global* symbol, not `My::Name`. Qualify definitions when the header declares
  them inside the namespace.
- ObjC class types mangle differently in C++ and ObjC++ translation units
  (`OBJC_CLASS` macro typedef vs real `@class`). Keep them out of cross-TU C++
  signatures; use `void*` type erasure for native providers.
- Static libraries must declare their own dependencies. Consumers that pull a
  library transitively otherwise hit link-order undefined symbols.
- Virtual dispatch through a base-class reference always lands on the
  most-derived override. If a host delegates a method to an owned backend, and
  the backend calls the "same" method back through a host interface reference,
  the call recurses forever (stack overflow). Break the cycle with a qualified
  base-class call in the host, plus a backend hook that runs only the
  backend-specific part. This bit every legacy app that is both host and
  backend owner; runtime compositions with plain hosts are immune.

### Backend/session lifecycle

- Keep a retired backend object alive until its replacement exists. Recovery
  paths still drain through the root's backend pointer; freeing early is a
  use-after-free.
- GUI platform bindings belong to the *window* lifetime. Unbind/rebind around
  window recreation; renderer ImGui backends also destroy platform windows on
  shutdown, so rebind around reconstruction even when the window is kept. A
  fresh binding can produce DeltaTime==0 on the first frame — clamp it.
- Session modules are re-bound by pointer (`BindSessionManagers`), never
  re-registered into the application module group: stale registrations
  accumulate and freed addresses get reused ("module already registered").
- The host's borrowed manager pointers (graphics/input) must be rebound after
  reconstruction: window resize/event routing dereferences them.
- The root tick must replicate the full application tick chain, including the
  backend's buffer swap and any headless pacing. Physics is fixed-step per
  tick; losing pacing changes simulation speed and diverges timed captures.
- Persistent RHIs must resync swapchain frame indices on recreate.
- Cross-backend switch captures are named after the *final* backend;
  intermediate backends must tolerate the requested format, and comparators
  should decode by magic, not extension.

### Platform API contracts

- Win32 `WM_PAINT` regenerates until the update region is validated
  (`ValidateRect`). An unbounded `PeekMessage(PM_REMOVE)` drain hangs.
- `SDL_VideoInit` — including the implicit init inside `SDL_CreateWindow` —
  resets GL attributes to defaults. Initialize the video subsystem *before*
  setting context attributes.
- SDL asset bootstrap must resolve paths against the executable directory
  (`SDL_GetBasePath`), not the CWD: the source asset tree lacks built
  shaders/materials.
- X11 `None` is a macro; it collides with enum/identifier names in shared
  headers.

### Verification methodology

- Cross-platform behavior cannot be fully verified locally; CI runs are
  *verification aids* during work steps, not acceptance. Acceptance is the
  formal deliverable gate (full roadmap + three-platform CI + PR).
- Software-rasterizer pairs (llvmpipe/lavapipe, MoltenVK/Metal) produce broad,
  low-amplitude rasterizer noise. Capture comparison should reject *structural*
  divergence (mean-delta, luminance, blank guards) rather than demand zero
  diff; edge-aliasing high-delta fractions vary run to run.
- Prefer the artifact that defines success (e.g. the produced PNG) over
  process exit codes, which can be unavailable depending on how the process
  was launched.
- When static analysis does not converge on a crash, capture a real stack
  (lldb/WinDbg) instead of guessing; bound the backtrace so diagnostics cannot
  hang the job.
- Each CI job is a fresh machine: workspaces carry *files*, not system state
  (installed SDKs, registry keys, machine PATH). Downstream jobs must recreate
  any system-level prerequisite they need.
- Pushing a new commit cancels the prior pipeline on that branch; hold pushes
  while a needed pipeline is running. Skip markers scan the entire head-commit
  message, subject and body.

## Part 2 — Current practical notes (re-verify when the dependency changes)

### CircleCI plan, open-source allowance, and topology

- **Depends on:** CircleCI Free plan + public repository (open-source credit
  allowance); current job split (build / regression / matrix). **Re-verify
  when:** plan changes, repository visibility changes, pricing changes, or job
  topology is restructured.
- Open-source allowance (official: circleci.com/open-source, /pricing FAQ,
  docs/guides/plans-pricing/credits): on the Free plan, **public repositories
  automatically get up to 400,000 credits/month for Linux/Arm/Docker OSS
  builds** — no application, and the allowance is not visible in the web app.
  macOS/Windows OSS builds and private repositories draw from the 30,000
  credits/month Free pool. OSS builds are not blocked when the credit balance
  is exhausted. This repository is public, so the allowance already applies.
  Larger OSS projects can contact CircleCI for paid-plan flexibility.
- The OSS allowance is a credit grant *within* the Free plan, not a plan-tier
  upgrade: plan-level feature limits still apply. **Job runtime caps are
  plan-level — 1h Free / 3h Performance / 5h Scale** (official support article
  "Max Runtime in CircleCI Server", 2026-02). OSS status does not raise them;
  the Windows split exists because the combined job hit the Free 1h cap, and
  each split job must stay under it.
- Free plan concurrency: 30 (Docker/Linux/Arm/Windows), **macOS = 1**. Larger
  Windows executor sizes (xlarge+) and macOS large require paid plans. Free
  plan also includes 5 active users/month and flaky-test detection for 5 tests.
- Workspaces fan out the build tree (`build/`, `build-release/`) plus generated
  in-source files (`config.h`) and the dependency-cache key file. The key file
  is shared because the key hashes tool versions (cmake/python/VS) that differ
  between jobs; recomputing it downstream caused cache misses.

### CircleCI Smarter Testing (testsuite)

- **Depends on:** circleci CLI 1.18.0 + testsuite extension 1.0.74712;
  `.circleci/test-suites.yml` schema. **Re-verify when:** CLI/extension
  upgrades, or the suite schema changes.
- The CLI executes suite `discover`/`run` commands through **bash on every
  platform, including Windows** (Git Bash), not the job's configured shell.
  Suite commands must be bash.
- Locally the extension must be installed (`circleci extension install
  testsuite`); CI images ship the CLI. Guard:
  `circleci testsuite --help || circleci extension install testsuite`.
- `doctor --json` prints progress lines before the JSON; strip them before
  parsing. Doctor validates by actually running tests.
- `options.store-test-results: true` replaces a `store_test_results` step —
  never both.
- Free features: static splitting (needs job `parallelism`), rerun-only-failed,
  test uploads. Credit-charged features (impact analysis, dynamic splitting,
  dynamic parallelism, auto-rerun) are opt-in and currently **not enabled**.
- **Pricing depends on:** current CircleCI pricing. **Re-verify when:** plan or
  pricing changes. Free plan includes 1,000 test atoms/month; overage is
  3,000 credits per 100,000 atoms. Local runs are always free. A full
  three-platform pipeline evaluates roughly 500 atoms (7 suite runs), so the
  Free allowance covers about two full pipelines/month; overage is ~15
  credits/pipeline — negligible next to compute cost, but still opt-in.

### Windows CI executor and Vulkan

- **Depends on:** `win/server-2022` image; LunarG SDK installed by choco
  (loader 1.4.363.1 at time of writing). **Re-verify when:** image or SDK
  version changes.
- The executor runs elevated in a service session with no enumerable
  interactive desktop: GDI window capture fails; use engine self-capture
  (`GEFS_SCREENSHOT_PATH`, glReadPixels→PNG). Debug assert dialogs hang runs
  (no desktop to dismiss) — they surface as step timeouts.
- The elevated Vulkan loader ignores `VK_ICD_FILENAMES`/`VK_DRIVER_FILES` and
  reads only the registry: register software ICDs under
  `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` (64-bit view).
- The SDK installer refreshes the System32 loader; the side-by-side SDK `Bin`
  ships no `vulkan-1.dll`. Jobs that need the loader must install the SDK
  themselves (see fresh-machine principle).
- PowerShell: with redirected output, `Process.ExitCode` can stay empty even
  after `WaitForExit(timeout)`; call the parameterless `WaitForExit()` or assert
  the produced artifact instead. `"$var:"` parses as a drive-qualified
  variable — write `"${var}:"`.

### Mesa software rasterizers

- **Depends on:** mesa-dist-win 26.3.0 (Windows); Ubuntu 24.04 mesa
  (llvmpipe/lavapipe) under Xvfb; SDL2. **Re-verify when:** mesa, SDL2, or the
  Xvfb/mesa package versions change — re-measure tolerances.
- Windows: deploy the lavapipe ICD DLL *with its sibling DLLs* next to the
  executable (app dir is first in the DLL search order); the ICD JSON's
  `library_path` must point at the deployed copy.
- Xvfb exposes only RGB visuals without alpha or multisample GLX configs, while
  SDL defaults request alpha=8 and the engine config requests 4x MSAA. Request
  alpha/depth/stencil 0 and no window MSAA (the engine renders into offscreen
  targets with their own depth/MSAA). A 4.3 core context works on llvmpipe.
- Linux lavapipe-vs-llvmpipe capture noise: ~30% differing pixels, mean delta
  ~4.6 over all pixels; high-delta fraction varies 2–3% run to run. Current
  guards: diff-fraction 0.40, high-delta 0.03, mean-delta 25 (over differing
  pixels) for startup/reconstruction/switch. Windows and macOS pairs keep
  high-delta 0.02 (measured green). Re-measure after any mesa upgrade.

### macOS CI

- **Depends on:** CircleCI macOS image (Apple Paravirtual Vulkan device);
  brew MoltenVK/validationlayers. **Re-verify when:** image or MoltenVK
  changes.
- `CAMetalLayerDrawable` "presented once" / "texture should not be called"
  warnings at the first GUI resize are pre-existing benign noise (present in
  green baselines), not a crash signature.
- DiagnosticReports may be empty on CI VMs; use lldb with a bounded backtrace
  (`thread backtrace -c 25`) — an unbounded one can time the step out.

### Local development environment (this snap host)

- **Depends on:** snap Ubuntu Core host, pip-installed ziglang/cmake/ninja,
  python3.14. **Re-verify when:** the host or toolchain layout changes.
- No system compiler: use the pip-installed zig wrappers; standalone
  Test/Runtime builds need `-DCMAKE_CXX_SCAN_FOR_MODULES=OFF` (zig ships no
  clang-scan-deps). zig c++ uses libc++; full platform code is verified only by
  CI.
- `curl` is not on PATH in fresh shells — use python urllib. No PIL/numpy;
  pure-python PNG decode (zlib + unfilter) works for capture analysis.
- The git endpoint intermittently rejects the token; the session's API push
  script (SHA-identical, handles merge commits, submodule gitlinks, empty
  commits, deletions via `--no-renames`) is the workaround. **Re-verify when:**
  endpoint behavior changes.

### Upstream tool behavior

- **Depends on:** ImGui version; CMake 4.x ctest. **Re-verify when:** either is
  upgraded.
- ImGui renderer backends (dx12/opengl3/metal) call
  `ImGui::DestroyPlatformWindows()` in `ShutdownMultiViewportSupport`; fresh
  platform binds can yield DeltaTime==0 (OSX/Win32 backends now clamp like
  SDL2).
- `ctest -N --show-only=json-v1` emits a `tests[].name` array; `-C` is accepted
  on single-config generators. The testsuite suites rely on both.