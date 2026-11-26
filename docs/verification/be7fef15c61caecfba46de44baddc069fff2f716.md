# Verification: be7fef15 (K-4c Windows consumer source-cache)

- **Revision:** `be7fef15` on `fix/ci-windows-source-cache`, base `master`
  `8be73f04`. Single config commit: `windows_test_setup` restores the
  weekly `source-v2` .git cache (pattern ported from build_windows), so the
  13 consumer jobs (regression, unit_debug, 11 matrix slots) no longer pay
  cold checkout + init submodules.
- **Pipelines:** iteration 1 workflow `692e30d0`: 14/16; two slots failed
  for reasons unrelated to the cache (job 5519 `choco vulkan-sdk` left
  System32 loader missing = infra class that killed 5411; job 5526
  d3d12->opengl->d3d12 switch genuinely incomplete = engine-flake class).
  Rerun from_failed workflow `177d7164`: **16/16 SUCCESS** — merge judged
  by rerun per agreement (PM seq875 thread).
- **Measured effect (rerun workflow, job wall + step clocks):**
  - init submodules: 315-425s (K-4b observation) -> **0.4-0.6m**
  - checkout: ~0.8-1m -> **0.1m**; restore repo 0.2-0.4m (cache-hit no-op)
  - windows_matrix_switch_viewer full job **19.0m** (three-run R1/R2/R3:
    32.9/27.9/29.9m) — back inside the original 26m line under the
    full-job caliber; regression 15.6m, unit_debug 12.5m.
- **Conclusion:** K-4b FAIL signal retired; per-slot fixed tax now
  ~8m (VM+tools+cache+attach) with attach 6.0-6.3m the remaining known
  item (release-zip decoupling parked to the vulkan-line backlog).
  35m recalibration line retained as conservative ceiling; observed
  ceiling now 19m.

## Limitations
- Cache is weekly + master-saved: first master run of each week pays the
  cold path in build_windows only; consumers then ride it.
- switch_viewer engine flake (1 occurrence in 5 runs across K-4b/K-4c)
  logged for the vulkan-switch-ux line; not CI-configuration.
