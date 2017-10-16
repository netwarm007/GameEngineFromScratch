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
find_library(XG_LIBRARY xg PATHS ${MYGE_EXTERNAL_LIBRARY_PATH})
target_link_libraries(Common ${XG_LIBRARY})

add_subdirectory(test)
