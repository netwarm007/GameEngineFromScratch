add_library(AssetIO AssetLoader.cpp)

add_library(SceneResources SceneResourceLoader.cpp)
target_link_libraries(SceneResources PUBLIC SceneGraph)
if(MSVC)
    target_link_libraries(SceneResources PRIVATE
        optimized ${ZLIB_LIBRARY} debug ${ZLIB_LIBRARY_DEBUG})
else()
    target_link_libraries(SceneResources PRIVATE ${ZLIB_LIBRARY})
endif()

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
target_link_libraries(Manager PUBLIC AssetIO SceneServices SceneResources)
