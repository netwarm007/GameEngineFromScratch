cmake_minimum_required (VERSION 2.8.11)

file(GLOB IMAGES
   "${PROJECT_SOURCE_DIR}/layersvt/images/*"
)
file(COPY ${IMAGES} DESTINATION ${CMAKE_BINARY_DIR}/layersvt/images)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR -DVK_USE_PLATFORM_WIN32_KHX -DWIN32_LEAN_AND_MEAN)
    set(DisplayServer Win32)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR -DVK_USE_PLATFORM_ANDROID_KHX)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if (BUILD_WSI_XCB_SUPPORT)
        add_definitions(-DVK_USE_PLATFORM_XCB_KHR -DVK_USE_PLATFORM_XCB_KHX)
        set(DisplayServer Xcb)
    endif()

    if (BUILD_WSI_XLIB_SUPPORT)
        if (NOT DisplayServer)
            set(DisplayServer Xlib)
        endif()
        add_definitions(-DVK_USE_PLATFORM_XLIB_KHR -DVK_USE_PLATFORM_XLIB_KHX)
    endif()

    if (BUILD_WSI_WAYLAND_SUPPORT)
       # TODO Add Wayland Support
       # add_definitions(-DVK_USE_PLATFORM_WAYLAND_KHR)
    endif()

    if (BUILD_WSI_MIR_SUPPORT)
       # TODO Add Mir Support
       # add_definitions(-DVK_USE_PLATFORM_MIR_KHR)
       # include_directories(${MIR_INCLUDE_DIR})
    endif()
else()
    message(FATAL_ERROR "Unsupported Platform!")
endif()

add_custom_target( generate_api_cpp DEPENDS api_dump.cpp )
add_custom_target( generate_api_h DEPENDS api_dump_text.h )
add_custom_target( generate_api_html_h DEPENDS api_dump_html.h )

set(LAYER_JSON_FILES
    VkLayer_api_dump
    VkLayer_monitor
    VkLayer_screenshot
    VkLayer_device_simulation
    )

set(VK_LAYER_RPATH /usr/lib/x86_64-linux-gnu/vulkan/layer:/usr/lib/i386-linux-gnu/vulkan/layer)
set(CMAKE_INSTALL_RPATH ${VK_LAYER_RPATH})

if (WIN32)
    if (NOT (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR))
        if (CMAKE_GENERATOR MATCHES "^Visual Studio.*")
            foreach (config_file ${LAYER_JSON_FILES})
                FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/windows/${config_file}.json src_json)
                FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>/${config_file}.json dst_json)
                add_custom_target(${config_file}-json ALL
                    COMMAND copy ${src_json} ${dst_json}
                    VERBATIM
                    )
                add_dependencies(${config_file}-json ${config_file})
            endforeach(config_file)
        else()
            foreach (config_file ${LAYER_JSON_FILES})
                FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/windows/${config_file}.json src_json)
                FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR}/${config_file}.json dst_json)
                add_custom_target(${config_file}-json ALL
                    COMMAND copy ${src_json} ${dst_json}
                    VERBATIM
                    )
                add_dependencies(${config_file}-json ${config_file})
            endforeach(config_file)
        endif()
    endif()
else()
    # extra setup for out-of-tree builds
    if (NOT (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR))
        foreach (config_file ${LAYER_JSON_FILES})
            add_custom_target(${config_file}-json ALL
                COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/linux/${config_file}.json
                VERBATIM
                )
             add_dependencies(${config_file}-json ${config_file})
        endforeach(config_file)
    endif()
    # Add targets for JSON file install on Linux.
    # Need to remove the "./" from the library path before installing to /etc.
    foreach (config_file ${LAYER_JSON_FILES})
        add_custom_target(${config_file}-staging-json ALL
            COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/staging-json
            COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/linux/${config_file}.json ${CMAKE_CURRENT_BINARY_DIR}/staging-json
            COMMAND sed -i -e "/\"library_path\":/s$./libVkLayer$libVkLayer$" ${CMAKE_CURRENT_BINARY_DIR}/staging-json/${config_file}.json
            VERBATIM
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/linux/${config_file}.json
            )
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/staging-json/${config_file}.json DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/vulkan/explicit_layer.d)
    endforeach(config_file)
    
endif()

if (WIN32)
    macro(add_vk_layer target)
    FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/VkLayer_${target}.def DEF_FILE)
    add_custom_target(copy-${target}-def-file ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DEF_FILE} VkLayer_${target}.def
        VERBATIM
    )
    add_library(VkLayer_${target} SHARED ${ARGN} VkLayer_${target}.def)
    add_dependencies(VkLayer_${target} generate_helper_files)
    target_link_Libraries(VkLayer_${target} VkLayer_utilsvt)
    add_dependencies(VkLayer_${target} generate_helper_files generate_api_cpp generate_api_h generate_api_html_h VkLayer_utilsvt)
    endmacro()
else()
    macro(add_vk_layer target)
    add_library(VkLayer_${target} SHARED ${ARGN})
    target_link_Libraries(VkLayer_${target} VkLayer_utilsvt)
    add_dependencies(VkLayer_${target} generate_helper_files generate_api_cpp generate_api_h generate_api_html_h VkLayer_utilsvt)
    set_target_properties(VkLayer_${target} PROPERTIES LINK_FLAGS "-Wl,-Bsymbolic")
    install(TARGETS VkLayer_${target} DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endmacro()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../loader
    ${CMAKE_CURRENT_SOURCE_DIR}/../layers
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/vulkan
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/../glslang/SPIRV
    ${JSONCPP_INCLUDE_DIR}
)

if (WIN32)
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_CRT_SECURE_NO_WARNINGS")
    set (CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -D_CRT_SECURE_NO_WARNINGS")

    # For VS 2015, which uses compiler version 1900, core_validation.cpp fails with too many objects
    # without either optimizations enabled, or setting the /bigobj compilation option.  Since
    # optimizations are enabled in a release build, this only affects the debug build.  For now,
    # enable /bigobj mode for all debug layer files. An alternative for the future is to split
    # large source files into multiple files which will also alleviate the compilation error.
    if (MSVC AND NOT (MSVC_VERSION LESS 1900))
        set (CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -D_CRT_SECURE_NO_WARNINGS /bigobj")
        set (CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG} -D_CRT_SECURE_NO_WARNINGS /bigobj")
    else()
        set (CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -D_CRT_SECURE_NO_WARNINGS")
        set (CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG} -D_CRT_SECURE_NO_WARNINGS")
    endif()
else()
    set (CMAKE_CXX_FLAGS "-std=c++11")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpointer-arith")
endif()

#VulkanTools layers
run_vk_xml_generate(api_dump_generator.py api_dump.cpp)
run_vk_xml_generate(api_dump_generator.py api_dump_text.h)
run_vk_xml_generate(api_dump_generator.py api_dump_html.h)

# Layer Utils Library
# For Windows, we use a static lib because the Windows loader has a fairly restrictive loader search
# path that can't be easily modified to point it to the same directory that contains the layers.
if (WIN32)
    add_library(VkLayer_utilsvt STATIC ../layers/vk_layer_config.cpp ../layers/vk_layer_extension_utils.cpp ../layers/vk_layer_utils.cpp ../layers/vk_format_utils.cpp)
else()
    add_library(VkLayer_utilsvt SHARED ../layers/vk_layer_config.cpp ../layers/vk_layer_extension_utils.cpp ../layers/vk_layer_utils.cpp ../layers/vk_format_utils.cpp)
    install(TARGETS VkLayer_utilsvt DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
add_dependencies(VkLayer_utilsvt generate_helper_files)

# VulkanTools layers
add_vk_layer(monitor monitor.cpp ../layers/vk_layer_table.cpp)
add_vk_layer(screenshot screenshot.cpp screenshot_parsing.h screenshot_parsing.cpp ../layers/vk_layer_table.cpp)
add_vk_layer(device_simulation device_simulation.cpp ../layers/vk_layer_table.cpp ${JSONCPP_SOURCE_DIR}/jsoncpp.cpp)
# generated
add_vk_layer(api_dump api_dump.cpp ../layers/vk_layer_table.cpp)

