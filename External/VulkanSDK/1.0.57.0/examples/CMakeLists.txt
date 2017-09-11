cmake_minimum_required(VERSION 2.8)
set(MAJOR "1")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

get_filename_component(INC_DIR "${CMAKE_SOURCE_DIR}/../x86_64/include" ABSOLUTE)
message("INC_DIR = ${INC_DIR}")
set(VK_INCLUDE_DIRS ${VK_INCLUDE_DIRS} "${INC_DIR}")

if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(COMMON_COMPILE_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers")
    set(COMMON_COMPILE_FLAGS "${COMMON_COMPILE_FLAGS} -fno-strict-aliasing -fno-builtin-memcmp")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 ${COMMON_COMPILE_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_COMPILE_FLAGS} -std=c++11 -fno-rtti")
    if (UNIX)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    endif()
endif()

get_filename_component(BIN_DIR "${CMAKE_SOURCE_DIR}/../x86_64/bin" ABSOLUTE)
message("BIN_DIR = ${BIN_DIR}")
find_program(GLSL_LANG_VALIDATOR glslangValidator PATHS ${BIN_DIR})

file(GLOB TEXTURES
  "${PROJECT_SOURCE_DIR}/*.ppm"
  )
file(COPY ${TEXTURES} DESTINATION ${CMAKE_BINARY_DIR})

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES")

option(BUILD_WSI_XCB_SUPPORT "Build XCB WSI support" ON)
option(BUILD_WSI_XLIB_SUPPORT "Build Xlib WSI support" ON)
option(BUILD_WSI_WAYLAND_SUPPORT "Build Wayland WSI support" ON)
option(BUILD_WSI_MIR_SUPPORT "Build Mir WSI support" ON)

if (NOT DEMOS_WSI_SELECTION)
    set(DEMOS_WSI_SELECTION "XCB")
endif()

if (DEMOS_WSI_SELECTION STREQUAL "XCB")
    if (NOT BUILD_WSI_XCB_SUPPORT)
	message( FATAL_ERROR "Selected XCB for demos build but not building Xcb support" )
    endif()
    find_package(XCB REQUIRED)
    include_directories(${XCB_INCLUDE_DIRS})
    link_libraries(${XCB_LIBRARIES})
    add_definitions(-DVK_USE_PLATFORM_XCB_KHR)
    set(DisplayServer Xcb)
elseif(DEMOS_WSI_SELECTION STREQUAL "XLIB")
    if (NOT BUILD_WSI_XLIB_SUPPORT)
	message( FATAL_ERROR "Selected XLIB for demos build but not building Xlib support" )
    endif()
    find_package(X11 REQUIRED)
    include_directories(${X11_INCLUDE_DIR})
    link_libraries(${X11_LIBRARIES})
    add_definitions(-DVK_USE_PLATFORM_XLIB_KHR)
    set(DisplayServer Xlib)
elseif(DEMOS_WSI_SELECTION STREQUAL "WAYLAND")
    if (NOT BUILD_WSI_WAYLAND_SUPPORT)
	message( FATAL_ERROR "Selected Wayland for demos build but not building Wayland support" )
    endif()
    find_package(Wayland REQUIRED)
    include_directories(${WAYLAND_CLIENT_INCLUDE_DIR})
    link_libraries(${WAYLAND_CLIENT_LIBRARIES})
    add_definitions(-DVK_USE_PLATFORM_WAYLAND_KHR)
    set(DisplayServer Wayland)
elseif(DEMOS_WSI_SELECTION STREQUAL "MIR")
    if (NOT BUILD_WSI_MIR_SUPPORT)
	message( FATAL_ERROR "Selected MIR for demos build but not building Mir support" )
    endif()
    find_package(Mir REQUIRED)
    add_definitions(-DVK_USE_PLATFORM_MIR_KHR)
    include_directories(${MIR_INCLUDE_DIR})
    # TODO - Add Mir support
    set(DisplayServer Mir)
else()
    message( FATAL_ERROR "Unrecognized value for DEMOS_WSI_SELECTION: ${DEMOS_WSI_SELECTION}" )
endif()

set(LIBRARIES "vulkan")

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/cube-vert.spv
   COMMAND ${GLSL_LANG_VALIDATOR} -s -V -o cube-vert.spv ${PROJECT_SOURCE_DIR}/cube.vert 
   DEPENDS cube.vert ${GLSL_LANG_VALIDATOR}
   )

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/cube-frag.spv
   COMMAND ${GLSL_LANG_VALIDATOR} -s -V -o ${CMAKE_BINARY_DIR}/cube-frag.spv ${PROJECT_SOURCE_DIR}/cube.frag 
   DEPENDS cube.frag ${GLSL_LANG_VALIDATOR}
   )


include_directories (
   ${VK_INCLUDE_DIRS}
   "${PROJECT_SOURCE_DIR}"
   )

if(WIN32)
   link_libraries(${XCB_LIBRARIES} vulkan-${MAJOR} png m)
elseif(UNIX)
   link_libraries(vulkan png m)
else()
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    get_filename_component(LIB_DIR "${CMAKE_SOURCE_DIR}/../source/lib" ABSOLUTE)
else()
    get_filename_component(LIB_DIR "${CMAKE_SOURCE_DIR}/../x86_64/lib" ABSOLUTE)
endif()
message("LIB_DIR = ${LIB_DIR}")
link_directories(${LIB_DIR})

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DEXTERNAL_SPV -DVK_USE_PLATFORM_XCB_KHR")

add_executable(vulkaninfo vulkaninfo.c)
target_link_libraries(vulkaninfo ${LIBRARIES})

add_executable(cube cube.c ${CMAKE_BINARY_DIR}/cube-vert.spv ${CMAKE_BINARY_DIR}/cube-frag.spv)
target_link_libraries(cube ${LIBRARIES})

add_executable(cubepp cube.cpp ${CMAKE_BINARY_DIR}/cube-vert.spv ${CMAKE_BINARY_DIR}/cube-frag.spv)
target_link_libraries(cubepp ${LIBRARIES})
