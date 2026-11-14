add_library(AssetIO AssetLoader.cpp)

add_library(SceneServices AnimationManager.cpp SceneManager.cpp)
target_link_libraries(SceneServices PUBLIC SceneGraph)

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
target_link_libraries(Manager PUBLIC AssetIO SceneServices)
