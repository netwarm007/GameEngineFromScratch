add_library(Common
Allocator.cpp
AssetLoader.cpp
BaseApplication.cpp
GraphicsManager.cpp
Image.cpp
MemoryManager.cpp
SceneManager.cpp
SceneObject.cpp
main.cpp
)
target_link_libraries(Common GeomMath)

if(WIN32)
    add_definitions(-DGUID_WINDOWS)
elseif(APPLE)
    add_definitions(-DGUID_CFUUID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
elseif(ANDROID)
    target_compile_definitions(Common PRIVATE GUID_ANDROID)
else()
    include_directories(${LIBUUID_INCLUDE_DIR})
    add_definitions(-DGUID_LIBUUID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
endif()


