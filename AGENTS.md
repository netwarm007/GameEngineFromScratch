# Instructions for AI-assisted work

## Project context

- This is a C++20 game engine built with CMake 4 or newer. Rendering backends,
  shaders, and asset processing are shared across multiple platforms.
- Follow the existing code style and platform guards. For rendering changes,
  check the affected backends and preserve their coordinate and shader semantics.
- Dependencies under `External/src` are Git submodules. Preserve existing local
  changes and ensure any new pinned commits are available from their configured
  remotes before proposing them.

## Workspace isolation

- Before starting any modification to this repository, an agent must create a
  dedicated Git worktree with its own branch (for example
  `git worktree add ../GameEngineFromScratch-<task> -b <branch> origin/master`)
  and make all edits, builds, and commits only inside that worktree. Do not
  modify the primary checkout or a worktree owned by another session. Remove
  the worktree (`git worktree remove`) and delete its branch only once the
  work's pull request is merged, or the work is abandoned. A passing CI run or
  an integration into an intermediate branch is not the cleanup trigger.

## Vulkan implementation

- For Vulkan backend work, read and follow the
  [Vulkan RHI implementation plan](docs/architecture/vulkan-rhi-implementation-plan.md).
  Check each implementation checkpoint against the plan and document justified
  deviations there. The upstream point-light shadow and ImGui lifecycle changes
  have been reviewed; integrate the latest master before implementation and
  preserve their contracts as detailed in the plan.

## Verification and acceptance

- Run focused local builds and tests for the change. Use `.circleci/config.yml`
  as the source of truth for platform-specific dependencies and commands.
  Local runs on the current platform are the fastest feedback; use
  `ctest -L fast` for the headless unit tests.
- CircleCI has two scopes. A normal push runs the full scope on all enabled
  platforms (every test, screenshots, coverage), which needs no API token and
  is what verification uses. For iteration, trigger the branch through the
  CircleCI API with `{"parameters": {"run_full": false, "platforms": "macos"}}`
  (token from the user's environment; never record it) to run only the quick
  scope on the listed platforms: ccache builds and `ctest -L fast`, with no
  screenshots, coverage, or packaging. A quick result never counts as
  full verification. Label new tests that need no window or GPU as `fast` (see
  `Test/CMakeLists.txt`). Triggering CI to verify multi-platform behavior during
  a work step is verification, not acceptance: this cross-platform project cannot
  be fully verified locally, so CI is the verification aid. The per-revision
  full-scope result is the verification gate for a commit. Acceptance (验收) is
  deliverable acceptance, the later culmination that normally requires all
  roadmap tasks complete, the three-platform CI passing, and a pull request.
- A project commit is verified only after the CircleCI pipeline for that exact
  revision completes successfully on every enabled platform: currently Windows,
  Linux, and macOS. Each platform must complete its configured build and tests.
  Update this list when the enabled CircleCI platforms change; disabled Android
  is not currently part of this verification gate.
- Local tests or a successful job on one platform do not replace the complete
  CircleCI result. Pending, canceled, skipped, unavailable, or failed jobs do
  not satisfy verification. Report the pipeline link, revision, each platform's
  result, and any blockers; do not claim verification while results are unknown.
- In this repository, verify the exact revision on all enabled CircleCI
  platforms and inspect required visual artifacts before opening a pull
  request; the PR is part of deliverable acceptance (roadmap tasks complete).
  Documentation-only changes covered by the exemption below do not require CI.
- Documentation-only commits are exempt from the CircleCI gate: do not run or
  wait for CircleCI, and put `[ci skip]` in the commit message so the push does
  not start a pipeline. A commit is documentation-only when every changed file
  is a Markdown file (`*.md`) or other prose documentation that is not read by
  a build, test, or script. Any change to source, build or CI configuration,
  scripts, assets, or tests is not exempt, even if the commit also changes
  documentation. A documentation-only commit does not verify a new source
  revision or supersede an existing verification record.
- After CircleCI passes on all enabled platforms for the exact revision, add
  `docs/verification/<full-tested-commit-sha>.md`. Include the pipeline link,
  each platform's job number and result, tested backends, relevant artifact
  observations, and limitations without implying untested configurations passed.
  Preserve other revisions' records; correct or supplement an existing record
  only with evidence for that same revision. Merges need their own CI results;
  passing parent revisions do not verify a merge revision.
- Known limitations of a roadmap-tracked effort (for example the memory
  allocator roadmap) belong in that roadmap's "Known limitations and follow-ups"
  section, with the checkpoint expected to address each. Verification records
  keep the per-revision limitations, but the roadmap is the single place that
  lists open ones. Add new limitations when found and remove or mark them
  resolved when fixed, in the same change.
- Do not update `README.md` for routine verification results or maintain a shared
  "latest verified revision" or result index. README links to the records
  directory and changes only for durable instructions, support, or behavior.
- For CircleCI investigation, use
  [the CircleCI troubleshooting skill](.github/skills/circleci-troubleshooting/SKILL.md).
- When monitoring a running CircleCI pipeline, run one background loop that
  checks the commit status about every 30 seconds and exits once every enabled
  platform has finished, then read its result on notification. For iteration it
  may stop at the first failure, but full-scope verification requires every
  enabled platform's terminal result. Do not poll with fixed multi-minute sleeps unless the user
  requests a specific interval. Report platforms that never appear as
  unknown, not passed. A push whose latest commit contains `[ci skip]` starts
  no pipeline, so push the revision to verify without it.
- If GitHub or CircleCI connectivity is unstable, check DNS resolution using a
  trusted DNS server as a diagnostic. Preserve hostname and TLS certificate
  verification, and do not change system DNS settings without authorization.

## Viewer UI and layout verification

- A successful build or screenshot step does not prove a panel is visible.
  Inspect the actual screenshot artifact for the affected platform and exact
  revision. Check overlap, clipping, and placement outside the main viewport
  before attributing a missing panel to scene loading or rendering.
- For video capture, target only the application, not the shared desktop.
  Window capture may return blank OpenGL frames; inspect an early sample and
  prefer the existing framebuffer readback path if needed. Verify decoded
  duration, frame count, and samples before, during, and after the action.
  A local file link is not a downloadable chat attachment; state delivery
  limitations rather than promising playback in the conversation.
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
