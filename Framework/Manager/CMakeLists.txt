add_library(AssetIO AssetLoader.cpp)

add_library(SceneResources SceneResourceLoader.cpp)
target_link_libraries(SceneResources PUBLIC SceneGraph PRIVATE ImageDecoders)

add_library(SceneServices AnimationManager.cpp SceneManager.cpp)
target_link_libraries(SceneServices PUBLIC SceneGraph)

add_library(SceneCamera SceneCamera.cpp)
target_link_libraries(SceneCamera PUBLIC SceneGraph)

add_library(SceneLighting SceneLighting.cpp)
target_link_libraries(SceneLighting PUBLIC SceneGraph)

add_library(Manager
        BaseApplication.cpp
        BlockAllocator.cpp
        DebugManager.cpp
        GraphicsManager.cpp
        InputManager.cpp
        MemoryManager.cpp
        StackAllocator.cpp
        PipelineStateManager.cpp
)
target_link_libraries(Manager PUBLIC AssetIO SceneServices SceneResources SceneCamera SceneLighting)
