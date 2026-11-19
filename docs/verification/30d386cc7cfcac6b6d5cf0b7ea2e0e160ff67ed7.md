# Deferred Release dependency replacement

Tested source revision: `30d386cc7cfcac6b6d5cf0b7ea2e0e160ff67ed7`.

## Exact-revision full CI

[CircleCI workflow 9f05e86a-32ef-4253-bf22-1b9f8a0cdb4a](https://app.circleci.com/pipelines/github/netwarm007/GameEngineFromScratch/9f05e86a-32ef-4253-bf22-1b9f8a0cdb4a/workflows/9f05e86a-32ef-4253-bf22-1b9f8a0cdb4a),
run `288de5c7-8c5b-49f3-9380-4207d92ca6d8`, used the full
`build-and-test` workflow on `agents/fixing-modles-directory-issue`. The
workflow succeeded in 56 minutes 14 seconds; it was not a quick/fast-only run.

| Platform | Tested backend | Job | Result |
| --- | --- | --- | --- |
| Windows | D3D12 / Win32 | 4769 | Debug build/tests and visual regressions passed before Release dependency replacement; Release build and all 78 CTest tests then passed |
| Linux | OpenGL / GLX / Xcb | 4766 | Debug build/tests, visual regressions and coverage passed before Release dependency replacement; Release build/tests then passed |
| macOS | Metal / Cocoa and Vulkan / MoltenVK | 4768 | Debug build/tests, visual regressions and platform Vulkan verification passed before Release dependency replacement; Release build/tests then passed |
| Linux Vulkan | Vulkan / lavapipe and native XCB | 4767 | Shader ABI, compute, resource, pipeline, graphics, GUI, swapchain and native application verification passed |

The tested ordering prevents Release shared libraries or DLLs from replacing
the external libraries used by later Debug application runs. Windows, Linux
and macOS Release dependency installation and Release engine tests now occur
only after each platform's existing Debug-dependent work.

## Limitations

Release receives compile-and-test coverage but not a separate visual
regression pass. Android remains disabled; Emscripten/WebAssembly, FreeBSD,
CUDA and OptiX are not accepted by this record. Software OpenGL and Vulkan do
not establish hardware-driver parity.

The documentation-only commit carrying this record uses `[ci skip]`. It does
not supersede the tested source revision or count as a separate source
verification.
