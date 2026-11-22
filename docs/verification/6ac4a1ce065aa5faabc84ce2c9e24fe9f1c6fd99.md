# Verification: 6ac4a1ce065aa5faabc84ce2c9e24fe9f1c6fd99

Change: `External/src/opengex` submodule updated to `6642cf7a8308dd4f965ad5967fcf7d213f79c698` (developed on branch `harden-oddl-parser` of `netwarm007/OpenGEX`; fast-forwarded into that fork's `master` after this run, so the pinned commit is unchanged): ODDL nesting depth limit (256) and unsigned integer overflow rejection. Adds an index-overflow case to `Test/Parser/OgexParserTest.cpp`.

Pipeline: https://app.circleci.com/workflow/70fae568-f7f7-477e-b942-890d8c58da93 (full scope, normal push)

| Job | Number | Result |
| --- | --- | --- |
| build_windows | 4887 | success |
| build_macos | 4889 | success |
| verify_vulkan_shader_abi | 4888 | success |
| build_linux | 4890 | success |

Tested backends: those configured in `.circleci/config.yml`; no backend-specific change.

Artifact observations: none inspected; no rendering or UI change.

Limitations:
- Locally the new test case was confirmed to fail without the submodule change and pass with it (g++, ASan/UBSan, Linux).
- On a synthetic 1M-vertex mesh, parsing took about 0.185s versus 0.138s before the parser rewrite (one platform, one file).
- CI ran while the submodule commit was only on the `harden-oddl-parser` branch; it is now also the fork's `master` head.
