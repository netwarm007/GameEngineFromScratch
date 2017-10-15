add_library(Common
Allocator.cpp
AssetLoader.cpp
BaseApplication.cpp
GraphicsManager.cpp
MemoryManager.cpp
SceneManager.cpp
main.cpp
)
target_link_libraries(Common GeomMath)

add_subdirectory(test)
