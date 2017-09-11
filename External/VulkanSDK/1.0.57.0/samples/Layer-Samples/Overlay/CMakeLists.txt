cmake_minimum_required(VERSION 2.8.11)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR -DWIN32_LEAN_AND_MEAN)
    set(DisplayServer Win32)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if (BUILD_WSI_XCB_SUPPORT)
        add_definitions(-DVK_USE_PLATFORM_XCB_KHR)
    endif()

    if (BUILD_WSI_XLIB_SUPPORT)
        add_definitions(-DVK_USE_PLATFORM_XLIB_KHR)
    endif()

    if (BUILD_WSI_WAYLAND_SUPPORT)
        add_definitions(-DVK_USE_PLATFORM_WAYLAND_KHR)
    endif()

    if (BUILD_WSI_MIR_SUPPORT)
        add_definitions(-DVK_USE_PLATFORM_MIR_KHR)
        include_directories(${MIR_INCLUDE_DIR})
    endif()
else()
    message(FATAL_ERROR "Unsupported Platform!")
endif()

set(SHADER_DIR "${PROJECT_SOURCE_DIR}/Layer-Samples/data")
set(SFILE overlay)

if(WIN32)
    set (MOVE_CMD "move")
else()
    set (MOVE_CMD "mv")
endif()

add_custom_command(OUTPUT ${SHADER_DIR}/${SFILE}-vert.spv
	COMMAND ${GLSLANG_VALIDATOR} -s -V ${SHADER_DIR}/${SFILE}.vert
	COMMAND ${MOVE_CMD} vert.spv ${SHADER_DIR}/${SFILE}-vert.spv
	DEPENDS ${SHADER_DIR}/${SFILE}.vert ${GLSLANG_VALIDATOR}
)
add_custom_command(OUTPUT ${SHADER_DIR}/${SFILE}-frag.spv
	COMMAND ${GLSLANG_VALIDATOR} -s -V ${SHADER_DIR}/${SFILE}.frag
	COMMAND ${MOVE_CMD} frag.spv ${SHADER_DIR}/${SFILE}-frag.spv
	DEPENDS ${SHADER_DIR}/${SFILE}.frag ${GLSLANG_VALIDATOR}
)

message(STATUS "Shader dir: ${SHADER_DIR}")
message(STATUS "Cxxflags: ${CMAKE_CXX_FLAGS}")

if(EXISTS "${CMAKE_SOURCE_DIR}/layers")
    set(LAYERS_LOC ${CMAKE_SOURCE_DIR})
    set(LAYER_UTILS_LIB VkLayer_utils)
else()
    set(LAYERS_LOC ${CMAKE_SOURCE_DIR}/../source)
    find_library(LAYER_UTILS_LIB NAMES VkLayer_utils
        HINTS "${CMAKE_SOURCE_DIR}/../${LIBSOURCE_DIR}" "${CMAKE_SOURCE_DIR}/../x86_64/lib" )
endif()

set(UTILDIR "${CMAKE_SOURCE_DIR}/Layer-Samples/utils")

add_library(VKLayer_overlay SHARED overlay.cpp
        ${LAYERS_LOC}/layers/vk_layer_table.cpp
        ${SHADER_DIR}/overlay-vert.spv
        ${SHADER_DIR}/overlay-frag.spv
        overlay.json)

if (WIN32)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_CRT_SECURE_NO_WARNINGS")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS")

    set_target_properties(VKLayer_overlay PROPERTIES LINK_FLAGS "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/VkLayer_overlay.def")
    add_custom_command(OUTPUT overlay.json
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/windows/overlay.json $<CONFIGURATION>/overlay.json
        VERBATIM)
else()
    set_target_properties(VKLayer_overlay PROPERTIES LINK_FLAGS "-Wl,-Bsymbolic")
    add_custom_command(OUTPUT overlay.json
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/linux/overlay.json overlay.json
        VERBATIM)
endif()

target_link_libraries(VKLayer_overlay ${LAYER_UTILS_LIB})
if(EXISTS "${CMAKE_SOURCE_DIR}/layers")
    add_dependencies(VKLayer_overlay generate_helper_files VkLayer_utils)
endif()
set_target_properties(VKLayer_overlay PROPERTIES CXX_FLAGS "-Wno-unused-function")

if(SDK_INCLUDE_PATH)
    include_directories(
            "${CMAKE_BINARY_DIR}/layers"
            "${CMAKE_BINARY_DIR}"
            "${LAYERS_LOC}/layers"
            "${LAYERS_LOC}/loader"
            "${UTILDIR}/stb"
            "${SDK_INCLUDE_PATH}"
            )
else()
    include_directories(
            "${CMAKE_BINARY_DIR}/layers"
            "${CMAKE_BINARY_DIR}"
            "${LAYERS_LOC}/layers"
            "${LAYERS_LOC}/loader"
            "${UTILDIR}/stb"
            )
endif()
