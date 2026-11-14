# Instructions for AI-assisted work

## Project context

- This is a C++20 game engine built with CMake 4 or newer. Rendering backends,
  shaders, and asset processing are shared across multiple platforms.
- Follow the existing code style and platform guards. For rendering changes,
  check the affected backends and preserve their coordinate and shader semantics.
- Dependencies under `External/src` are Git submodules. Preserve existing local
  changes and ensure any new pinned commits are available from their configured
  remotes before proposing them.

## Verification and acceptance

- Run focused local builds and tests for the change. Use `.circleci/config.yml`
  as the source of truth for platform-specific dependencies and commands.
- A project commit is accepted only after the CircleCI pipeline for that exact
  revision completes successfully on every enabled platform: currently Windows,
  Linux, and macOS. Each platform must complete its configured build and tests.
  Update this list when the enabled CircleCI platforms change; disabled Android
  is not currently part of this acceptance gate.
- Local tests or a successful job on one platform do not replace the complete
  CircleCI result. Pending, canceled, skipped, unavailable, or failed jobs do
  not satisfy acceptance. Report the pipeline link, revision, each platform's
  result, and any blockers; do not claim acceptance while results are unknown.
- Documentation-only commits are exempt from the CircleCI gate: do not run or
  wait for CircleCI, and put `[ci skip]` in the commit message so the push does
  not start a pipeline. A commit is documentation-only when every changed file
  is a Markdown file (`*.md`) or other prose documentation that is not read by
  a build, test, or script. Any change to source, build or CI configuration,
  scripts, assets, or tests is not exempt, even if the commit also changes
  documentation. A documentation-only commit keeps the last verified revision;
  it does not replace it.
- After CircleCI passes on all enabled platforms, update relevant sections of
  `README.md` to reflect the latest verified results. Identify the tested
  revision and platform coverage without implying untested configurations passed.
- For CircleCI investigation, use
  [the CircleCI troubleshooting skill](.github/skills/circleci-troubleshooting/SKILL.md).
- When monitoring a running CircleCI pipeline, check its status every five
  minutes by default unless the user requests a different interval.
- If GitHub or CircleCI connectivity is unstable, check DNS resolution using a
  trusted DNS server as a diagnostic. Preserve hostname and TLS certificate
  verification, and do not change system DNS settings without authorization.

## Viewer UI and layout verification

- A successful build or screenshot step does not prove a panel is visible.
  Inspect the actual screenshot artifact for the affected platform and exact
  revision. Check overlap, clipping, and placement outside the main viewport
  before attributing a missing panel to scene loading or rendering.
- With ImGui multi-viewports enabled, window positions are desktop coordinates.
  Derive initial placement from the main viewport's work position and size,
  not fixed coordinates or framebuffer pixel dimensions.
- Default Viewer panels use docking. Initialize defaults only when no saved
  layout exists; preserve user rearrangements and floating windows. Rebuilding
  the dock tree every frame destroys those choices. The Debug menu provides
  an explicit reset action for restoring defaults, including older layouts.
- Use `GuiSubPassLayoutTest` for headless layout regression checks. Cover
  docking, non-overlap, viewport containment, resizing, non-zero viewport
  origins, Retina framebuffer scaling, and saved layout preservation.
  Confirm the regression fails without the fix, then inspect a real Viewer
  screenshot; headless tests do not replace backend rendering verification.

## Safety

- Preserve unrelated changes. Do not commit, push, create remote repositories,
  or trigger pipelines unless the user has authorized the action.
- Never record or disclose credentials or raw sensitive CI output. If Git LFS
  authentication is required, ask the user to enter credentials through the
  appropriate system interface.
