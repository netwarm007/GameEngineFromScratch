# M3.1: render-session contract and persistent runtime root

## Tested revision and scope

- Tested source: `252ac2abb32ab8db651adf4cdcce57a308043cbb`.
- Branch: `feat/runtime-rhi-composition`.
- Full CircleCI [pipeline 1126](https://app.circleci.com/pipelines/github/netwarm007/GameEngineFromScratch/1126)
  ([run](https://app.circleci.com/pipeline/8d68ccd9-cfcb-49d6-9f80-719d0312c07e)).
- Workflow: `build-and-test`, `89c5a947-aa46-4d60-822a-fd69e60aa8a4`, successful.

This is the verification record for the M3.1 work step in the
[runtime RHI roadmap](../architecture/runtime-rhi-switching-roadmap.md). It
covers the headless session contract only:

- [RenderSession.hpp](../../Framework/Manager/RenderSession.hpp):
  `RenderSessionSlot` with per-cycle borrowed registration, forward
  activation with reverse rollback and an owner GPU-drain hook, reverse
  retirement clearing registrations, and a session generation independent of
  scene revision (published on successful activation and on retirement of an
  active session; failed activation and repeated retirement publish nothing).
- [RuntimeRoot.hpp](../../Framework/Manager/RuntimeRoot.hpp) /
  [RuntimeRoot.cpp](../../Framework/Manager/RuntimeRoot.cpp): persistent
  CPU-service module group, one session slot, and the engine allocator stack
  owned at the root (non-fatal unconfigured fallback, hook cleared on
  destruction).
- [RenderSessionTest.cpp](../../Test/Runtime/RenderSessionTest.cpp): `fast`
  regression covering repeated session cycles with persistent services
  untouched, failure rollback/retry with the drain hook, generation
  semantics, caller-composed interleaved frame order across replacement, and
  allocator-stack ownership across session cycles and root destruction.

Applications keep their existing startup paths; no native switching or
reconstruction behavior is claimed. Native wiring is M3.4.

This was a full normal-push run. Per the agreed vocabulary this is a
**verification** record (per-revision full-CI evidence), not deliverable
acceptance.

## Platform results

| Job | Number | Result | Relevant coverage |
| --- | --- | --- | --- |
| `build_linux` | 5011 | Success | Standalone `Test/Runtime` suite incl. `RenderSessionTest`; full build `fast` suite links `Manager`. |
| `build_macos` | 5012 | Success | Same standalone and full-build coverage. |
| `build_windows` | 5013 | Success | Same standalone and full-build coverage (MSVC). |

The new regression runs in both the standalone `Test/Runtime` build (allocator
closure compiled directly) and the full build (`Manager` link), on all three
platforms, in Debug. Existing `RuntimeModuleLifecycleTest` and `RhiStartupTest`
remain green; the full native startup matrices are unchanged by this
headless-only step.

## Local pre-check

The standalone suite was also built and run locally (clang-based zig c++
0.17.1, libc++, Debug): 3/3 tests passed before the push. One test-side
expectation was corrected during development (the replacement activation's
`init` event precedes the frame assertion); no production change resulted
from it.