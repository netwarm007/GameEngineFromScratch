add_library(Common
Allocator.cpp
BaseApplication.cpp
AssetLoader.cpp
GraphicsManager.cpp
MemoryManager.cpp
main.cpp
)
target_link_libraries(Common GeomMath)

add_subdirectory(test)
