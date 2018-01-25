add_library(Common
Allocator.cpp
AssetLoader.cpp
BaseApplication.cpp
GameLogic.cpp
GraphicsManager.cpp
InputManager.cpp
Image.cpp
MemoryManager.cpp
PhysicsManager.cpp
Scene.cpp
SceneManager.cpp
SceneObject.cpp
main.cpp
)
target_link_libraries(Common 
        GeomMath
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
        ${XG_LIBRARY} 
        ${ZLIB_LIBRARY}
        ${BULLET_DYNAMICS_LIBRARY} 
        ${BULLET_COLLISION_LIBRARY} 
        ${BULLET_LINEARMATH_LIBRARY})

if(WIN32)
    add_definitions(-DGUID_WINDOWS)
elseif(APPLE)
    add_definitions(-DGUID_CFUUID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
elseif(ANDROID)
    target_compile_definitions(Common PRIVATE GUID_ANDROID)
elseif(ORBIS)
    # nothing here
else(WIN32)
    find_package(Libuuid REQUIRED)
    if (NOT LIBUUID_FOUND)
        message(FATAL_ERROR
            "You might need to run 'sudo apt-get install uuid-dev' or similar")
    endif()
    include_directories(${LIBUUID_INCLUDE_DIR})
    add_definitions(-DGUID_LIBUUID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
endif()


