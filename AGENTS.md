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
- For CircleCI investigation, use
  [the CircleCI troubleshooting skill](.github/skills/circleci-troubleshooting/SKILL.md).

## Safety

- Preserve unrelated changes. Do not commit, push, create remote repositories,
  or trigger pipelines unless the user has authorized the action.
- Never record or disclose credentials or raw sensitive CI output. If Git LFS
  authentication is required, ask the user to enter credentials through the
  appropriate system interface.
