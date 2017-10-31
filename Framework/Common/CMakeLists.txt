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
find_library(XG_LIBRARY xg PATHS ${MYGE_EXTERNAL_LIBRARY_PATH})
find_library(OPENDDL_LIBRARY OpenDDL PATHS ${MYGE_EXTERNAL_LIBRARY_PATH})
find_library(OPENGEX_LIBRARY OpenGEX PATHS ${MYGE_EXTERNAL_LIBRARY_PATH})

if(WIN32)
    add_definitions(-DGUID_WINDOWS)
elseif(APPLE)
    find_library(CFLIB CoreFoundation)
    target_link_libraries(Common ${CFLIB})
    add_definitions(-DGUID_CFUUID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
elseif(ANDROID)
    target_compile_definitions(Common PRIVATE GUID_ANDROID)
else()
    find_package(Libuuid REQUIRED)
    if (NOT LIBUUID_FOUND)
        message(FATAL_ERROR
            "You might need to run 'sudo apt-get install uuid-dev' or similar")
    endif()
    include_directories(${LIBUUID_INCLUDE_DIR})
    target_link_libraries(Common ${LIBUUID_LIBRARY})
    add_definitions(-DGUID_LIBUUID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
endif()


