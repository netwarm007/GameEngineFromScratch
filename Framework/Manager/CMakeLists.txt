add_library(AssetIO AssetLoader.cpp)

add_library(SceneResources SceneResourceLoader.cpp)
target_link_libraries(SceneResources PUBLIC SceneGraph PRIVATE ImageDecoders)

add_library(SceneServices AnimationManager.cpp SceneManager.cpp)
target_link_libraries(SceneServices PUBLIC SceneGraph)

add_library(SceneCamera SceneCamera.cpp)
target_link_libraries(SceneCamera PUBLIC SceneGraph)

add_library(SceneLighting SceneLighting.cpp)
target_link_libraries(SceneLighting PUBLIC SceneGraph)

add_library(RenderServices GraphicsManager.cpp)
target_link_libraries(RenderServices PUBLIC SceneCamera SceneLighting
    PRIVATE DrawPass DrawSubPass DispatchPass)

add_library(Manager
        BaseApplication.cpp
        BlockAllocator.cpp
        BuddyAllocator.cpp
        DebugManager.cpp
        FrameAllocator.cpp
        InputManager.cpp
        LinearAllocator.cpp
        MemoryManager.cpp
        StackAllocator.cpp
        TlsfAllocator.cpp
        TrackingAllocator.cpp
        PipelineStateManager.cpp
        RingAllocator.cpp
        SizeClassAllocator.cpp
        SlabAllocator.cpp
)
target_link_libraries(Manager PUBLIC AssetIO SceneServices SceneResources RenderServices)
