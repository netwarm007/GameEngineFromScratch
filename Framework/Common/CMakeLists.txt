add_library(Common
        AnimationManager.cpp
        AssetLoader.cpp
        BaseApplication.cpp
        BlockAllocator.cpp
        DebugManager.cpp
        GraphicsManager.cpp
        InputManager.cpp
        Image.cpp
        MemoryManager.cpp
        SceneManager.cpp
        StackAllocator.cpp
        PipelineStateManager.cpp
        main.cpp
)

find_package(Threads REQUIRED)

target_link_libraries(Common
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
        ${XG_LIBRARY} 
        ${ZLIB_LIBRARY}
        Threads::Threads
)

__add_xg_platform_dependencies(Common)

