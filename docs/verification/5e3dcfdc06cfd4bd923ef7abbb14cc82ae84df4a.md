# Splash bouncing ball verification (merged with master)

CircleCI [pipeline 918](https://app.circleci.com/pipelines/github/netwarm007/GameEngineFromScratch/918)
(workflow `e0d16b21-e52e-4e65-8387-59aed8969998`) passed the configured builds
and tests for merge revision `5e3dcfdc06cfd4bd923ef7abbb14cc82ae84df4a`, which
merges master (`ba87d847`) into the Splash ball branch. The only conflict was
`Viewer/main.cpp`: the Viewer keeps `BulletPhysicsManager` and adopts master's
`SceneResourceLoader` wiring.

| Platform | Job | Result |
| --- | --- | --- |
| Windows | 4306 | Build, tests, Viewer and Billiard screenshots passed |
| Linux | 4305 | Build, tests, Viewer and Billiard screenshots passed |
| macOS | 4307 | Build, tests, Viewer and Billiard screenshots passed |

Tests included `PhysicsSceneIntegrationTest::TestSplashBouncingBall` and the
extended `OgexParserTest`. The Viewer uses Bullet on all three platforms.

Exact-revision screenshots were inspected:

- `viewer-render.png` (all platforms): the ball is visible on the text, panels
  are docked without overlap or clipping, and the scene is inside the main
  viewport. Windows has a mouse-hover texture tooltip that does not cover the
  ball. The macOS floor looks greener than on Linux/Windows; this was not
  investigated and is not attributed to this change.
- `billiard-after.png` (all platforms): table, ball cluster and cues are
  visible, with no regression.

Limitations: a still image shows one instant; bouncing is evidenced by the
integration test, not the screenshots. The ball position differs per platform
because the trajectory is chaotic. Only CI exercised the Viewer with LFS
textures.
