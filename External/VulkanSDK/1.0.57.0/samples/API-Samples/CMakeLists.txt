cmake_minimum_required(VERSION 2.8.11)

# If ANDROID is ON, turn on cross-compiling for it
if(ANDROID)
    set(CMAKE_SYSTEM_NAME "Android")
    set(CMAKE_SYSTEM_VERSION "7")
endif()

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

set(SAMPLES_DATA_DIR ${SAMPLES_DATA_DIR} "${CMAKE_SOURCE_DIR}/API-Samples/data")
set(SHADER_FILES ${SHADER_FILES} "")
include_directories( ${SAMPLES_DATA_DIR} ${GLSLANG_SPIRV_INCLUDE_DIR} ${GLMINC_PREFIX})

# Additional includes for spirv-tools
include_directories(${SPIRV_TOOLS_INCLUDE_DIR})

# function to build a single-source-file sample that uses glsl shaders;
#   - assumes SAMPLE_GLSL_VERT_SHADERS and SAMPLE_GLSL_FRAG_SHADERS is set to
#     the list of shader files of that type, with .vert and .frag omitted
#   - uses glslangValidator to convert glsl to spirv
function(sampleExtGLSLShaders SNAME)
    set(SAMPLE_NAME ${SNAME})
    set(SHADER_DIR ${SAMPLES_DATA_DIR}/${VULKAN_VERSION})
    foreach(SFILE ${SAMPLE_GLSL_VERT_SHADERS})
        add_custom_command(OUTPUT ${SHADER_DIR}/${SFILE}-vert.spv
            COMMAND ${GLSLANG_VALIDATOR} -s -V ${SHADER_DIR}/${SFILE}.vert
            COMMAND ${MOVE_CMD} vert.spv ${SHADER_DIR}/${SFILE}-vert.spv
            DEPENDS ${SHADER_DIR}/${SFILE}.vert ${GLSLANG_VALIDATOR}
        )
        set(SHADER_FILES ${SHADER_FILES} ${SHADER_DIR}/${SFILE}-vert.spv)
    endforeach(SFILE)
    foreach(SFILE ${SAMPLE_GLSL_FRAG_SHADERS})
        add_custom_command(OUTPUT ${SHADER_DIR}/${SFILE}-frag.spv
            COMMAND ${GLSLANG_VALIDATOR} -s -V ${SHADER_DIR}/${SFILE}.frag
            COMMAND ${MOVE_CMD} frag.spv ${SHADER_DIR}/${SFILE}-frag.spv
            DEPENDS ${SHADER_DIR}/${SFILE}.frag ${GLSLANG_VALIDATOR}
        )
        set(SHADER_FILES ${SHADER_FILES} ${SHADER_DIR}/${SFILE}-frag.spv)
    endforeach(SFILE)
    add_executable(${SAMPLE_NAME} ${SAMPLE_NAME}/${SAMPLE_NAME}.cpp ${SHADER_FILES})
    target_link_libraries(${SAMPLE_NAME} ${VULKAN_LOADER} ${UTILS_NAME} ${GLSLANG_LIBRARIES} ${PTHREAD})
endfunction(sampleExtGLSLShaders)

# function to build a single-source-file sample that uses spirv shaders
#   - assumes SAMPLE_SPIRV_SHADERS is set to list to list of spirv shsders
#     used by the sample, with the .spv omitted
function(sampleExtSPIRVShaders SNAME)
    set(SAMPLE_NAME ${SNAME})
    set(SHADER_DIR ${SAMPLES_DATA_DIR}/${VULKAN_VERSION})
    foreach(SFILE ${SAMPLE_SPIRV_SHADERS})
        set(SHADER_FILES ${SHADER_FILES} ${SHADER_DIR}/${SFILE}.spv)
    endforeach(SFILE)
    add_executable(${SAMPLE_NAME} ${SAMPLE_NAME}/${SAMPLE_NAME}.cpp ${SHADER_FILES})
    target_link_libraries(${SAMPLE_NAME} ${VULKAN_LOADER} ${UTILS_NAME} ${GLSLANG_LIBRARIES} ${PTHREAD})
endfunction(sampleExtSPIRVShaders)

# function to build a simple single-source-file sample
#   - assumes S_TARGETS set to list of build targets
function(sampleWithSingleFile)
    foreach(TARG ${S_TARGETS})
        set(SAMPLE_NAME ${TARG})
        if(ANDROID)
            # Generate AndroidStudio project files.
            MESSAGE("Generating AndroidStudio project:" ${SAMPLE_NAME})
            # Replace "-" to "_", which is invalid in a package name.
            STRING(REPLACE "-" "_" PACKAGE_NAME ${SAMPLE_NAME})
            STRING(REGEX MATCH "^[0-9]" LEADING_DIGIT "${PACKAGE_NAME}")
            if(NOT LEADING_DIGIT STREQUAL "")
                set(PACKAGE_NAME "T${PACKAGE_NAME}")
            endif()
            set (PACKAGE_NAME vulkan.samples.${PACKAGE_NAME})
            set (SAMPLE_DESCRIPTION "Vulkan sample: ${SAMPLE_NAME}")
            file(COPY ./android/project_template/ DESTINATION ./android/${SAMPLE_NAME}/)

            # Clear flags.
            set (OTHER_FLAGS "")
            set (ASSET_DIR "")
            set (EXTRA_PERMISSIONS "")
            set (SHADERC_LINKAGE "")
            set (VALIDATIONLAYER_PACKING "")

            # Add extra setting for some samples.
            # shaderc usage.
            set (SHADERC_LINKAGE "library \"shaderc\"         linkage \"static\"")

            # Include resource files.
            set (SAMPLE_WITH_RESOURCES
             draw_textured_cube init_texture immutable_sampler memory_barriers multiple_sets pipeline_cache
             pipeline_derivative secondary_command_buffer separate_image_sampler spirv_assembly
             spirv_specialization template)
            if (";${SAMPLE_WITH_RESOURCES};" MATCHES ";${SAMPLE_NAME};")
                set (ASSET_DIR "sourceSets.main.assets.srcDirs '../../data'")
            endif()

            # Add external storage access permission.
            set (SAMPLE_WITH_EXTERNALSTORAGEACCESS pipeline_cache)
            if (";${SAMPLE_WITH_EXTERNALSTORAGEACCESS};" MATCHES ";${SAMPLE_NAME};")
                set (EXTRA_PERMISSIONS "<uses-permission android:name=\"android.permission.WRITE_EXTERNAL_STORAGE\"/>")
            endif()

            # Add validation layer settings.
            set (SAMPLE_WITH_TOOLS create_debug_report_callback enable_validation_with_callback)
            if (";${SAMPLE_WITH_TOOLS};" MATCHES ";${SAMPLE_NAME};")
                set (VALIDATIONLAYER_PACKING "sourceSets.main.jniLibs.srcDirs file(ndkDir).absolutePath +\n\t\t\t\t'/sources/third_party/vulkan/src/build-android/jniLibs'")
            endif()

            # Cook template files.
            set (TEMPLATE_FILES
                android/${SAMPLE_NAME}/src/main/res/values/strings.xml
                android/${SAMPLE_NAME}/src/main/AndroidManifest.xml
                android/${SAMPLE_NAME}/build.gradle
                android/${SAMPLE_NAME}/README.md
            )
            foreach(TEMPLATE_FILE ${TEMPLATE_FILES})
                configure_file(${TEMPLATE_FILE} ${TEMPLATE_FILE} @ONLY)
            endforeach(TEMPLATE_FILE)
            SET(SETTINGS_GRADLE "${SETTINGS_GRADLE}include ':${SAMPLE_NAME}'\n")
        elseif(UNIX)
            add_executable(${SAMPLE_NAME} ${SAMPLE_NAME}/${SAMPLE_NAME}.cpp)
            target_link_libraries(${SAMPLE_NAME} ${UTILS_NAME} ${GLSLANG_LIBRARIES} ${XCB_LIBRARIES} ${VULKAN_LOADER} ${PTHREAD} ${SPIRV_TOOLS_LIBRARIES})
        else()
            add_executable(${SAMPLE_NAME} WIN32 ${SAMPLE_NAME}/${SAMPLE_NAME}.cpp)
            target_link_libraries(${SAMPLE_NAME} ${UTILS_NAME} ${GLSLANG_LIBRARIES} ${VULKAN_LOADER} ${WINLIBS} ${SPIRV_TOOLS_LIBRARIES})
        endif()
    endforeach(TARG)

    if(ANDROID)
        #generate Settings.gradle
        configure_file(android/settings.gradle.in android/settings.gradle @ONLY)
    endif()
endfunction(sampleWithSingleFile)

# function to build a simple single-source-file sample
#   - assumes that code is standalone and doesn't depend on utilitites
#   - assumes S_TARGETS set to list of build targets
function(sampleInlineGLSLShaders)
    foreach(TARG ${S_TARGETS})
        set(SAMPLE_NAME ${TARG})
        if(UNIX)
            add_executable(${SAMPLE_NAME} ${SAMPLE_NAME}/${SAMPLE_NAME}.cpp)
            target_link_libraries(${SAMPLE_NAME} ${GLSLANG_LIBRARIES} ${XCB_LIBRARIES} ${VULKAN_LOADER} ${PTHREAD})
        else()
            add_executable(${SAMPLE_NAME} WIN32 ${SAMPLE_NAME}/${SAMPLE_NAME}.cpp)
            target_link_libraries(${SAMPLE_NAME} ${GLSLANG_LIBRARIES} ${VULKAN_LOADER} ${WINLIBS})
        endif()
    endforeach(TARG)
endfunction(sampleInlineGLSLShaders)

if(WIN32)
    set (MOVE_CMD "move")
    set (VULKAN_LOADER_NAME "${API_LOWERCASE}-${MAJOR}")
else()
    set (MOVE_CMD "mv")
    set (PTHREAD "pthread")
    set (VULKAN_LOADER_NAME "${API_LOWERCASE}")
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/layers")
    set (VULKAN_LOADER ${VULKAN_LOADER_NAME})
else()
    find_library(VULKAN_LOADER NAMES ${VULKAN_LOADER_NAME}
    HINTS "${CMAKE_SOURCE_DIR}/../${LIBSOURCE_DIR}" "${CMAKE_SOURCE_DIR}/../x86_64/lib" )
endif()

set (LIBGLM_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/libs)

if(NOT WIN32 AND NOT ANDROID)
    if(SDK_INCLUDE_PATH)
        include_directories(
            ${XCB_INCLUDE_DIRS}
            ${SDK_INCLUDE_PATH}
        )
    else()
        include_directories(
            ${XCB_INCLUDE_DIRS}
        )
    endif()
    link_libraries(${XCB_LIBRARIES} ${VULKAN_LOADER} m )
endif()
if(WIN32)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES")

    # If MSVC, disable some signed/unsigned mismatch warnings.
    if (MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4267")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267")
    endif()

else()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
endif()

if (NOT WIN32)
    # extra setup for out-of-tree builds
    if (NOT (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR))
        add_custom_target(samples-binary-dir-symlinks ALL
            COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/run_all_samples.sh
            COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/get-short-descripts.sh
            VERBATIM
            )
    endif()
else()
    if (NOT (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR))
        FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lunarg.ppm LUNARG)
        FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/run_all_samples.ps1 RUN_ALL)
        add_custom_target(samples-binary-dir-symlinks ALL
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RUN_ALL} run_all_samples.ps1
            VERBATIM
            )
    endif()
endif()


#
# START BUILDING SAMPLES HERE
#

# Generate Android project.
option(ANDROID OFF)

# simple one file sample targets, no additional files
set (S_TARGETS
    01-init_instance 02-enumerate_devices 03-init_device 04-init_command_buffer
    05-init_swapchain 06-init_depth_buffer 07-init_uniform_buffer
    08-init_pipeline_layout 09-init_descriptor_set 10-init_render_pass
    11-init_shaders 12-init_frame_buffers 13-init_vertex_buffer 14-init_pipeline 15-draw_cube
    enumerate_devices_adv init_texture multithreaded_command_buffers
    draw_textured_cube dynamic_uniform texel_buffer events
    instance_layer_properties instance_extension_properties
    instance_layer_extension_properties create_debug_report_callback
    enable_validation_with_callback multiple_sets
    copy_blit_image template separate_image_sampler input_attachment
    occlusion_query pipeline_cache pipeline_derivative push_descriptors
    immutable_sampler push_constants draw_subpasses secondary_command_buffer
    spirv_assembly spirv_specialization memory_barriers)
sampleWithSingleFile()

if (NOT ANDROID)
    add_subdirectory(utils)
endif()

# targets that use spv shaders

# usespirvshader
# set(SAMPLE_SPIRV_SHADERS spirvshader-vert spirvshader-frag)
# sampleExtSPIRVShaders()

# targets that require glsl to spv shader conversion

# useglslshader
# set(SAMPLE_GLSL_FRAG_SHADERS glslshader)
# set(SAMPLE_GLSL_VERT_SHADERS glslshader)
# sampleExtGLSLShaders()
