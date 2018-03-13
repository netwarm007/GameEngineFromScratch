add_library(PlatformInterface
                OrbisApplication.cpp 
                OrbisGalicHeapAllocator.cpp
                OrbisMemoryManager.cpp
                OrbisOnionHeapAllocator.cpp
)

add_library(MyGameEngine
                OrbisApplicationConfig.cpp
    )
target_link_libraries(MyGameEngine
                        Common 
                        OrbisGnm
                        PlatformInterface
)

