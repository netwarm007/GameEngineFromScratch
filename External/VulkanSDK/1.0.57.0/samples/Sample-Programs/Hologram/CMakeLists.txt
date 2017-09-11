find_package(PythonInterp 3 REQUIRED)
find_package(Threads REQUIRED)

macro(generate_dispatch_table out)
    add_custom_command(OUTPUT ${out}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate-dispatch-table ${out}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate-dispatch-table
        )
endmacro()

macro(glsl_to_spirv src)
    add_custom_command(OUTPUT ${src}.h
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/glsl-to-spirv ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${src}.h ${GLSLANG_VALIDATOR}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/glsl-to-spirv ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${GLSLANG_VALIDATOR}
        )
endmacro()

generate_dispatch_table(HelpersDispatchTable.h)
generate_dispatch_table(HelpersDispatchTable.cpp)
glsl_to_spirv(Hologram.frag)
glsl_to_spirv(Hologram.vert)
glsl_to_spirv(Hologram.push_constant.vert)

set(sources
    Game.h
    Helpers.h
    HelpersDispatchTable.cpp
    HelpersDispatchTable.h
    Hologram.cpp
    Hologram.h
    Hologram.frag.h
    Hologram.vert.h
    Hologram.push_constant.vert.h
    Main.cpp
    Meshes.cpp
    Meshes.h
    Meshes.teapot.h
    Simulation.cpp
    Simulation.h
    Shell.cpp
    Shell.h
    )

set(definitions
    PRIVATE -DVK_NO_PROTOTYPES
    PRIVATE -DGLM_FORCE_RADIANS)

if(SDK_INCLUDE_PATH)
    set(includes
        PRIVATE ${GLMINC_PREFIX}
        PRIVATE ${SDK_INCLUDE_PATH}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
else()
    set(includes
        PRIVATE ${GLMINC_PREFIX}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endif()

set(libraries PRIVATE ${CMAKE_THREAD_LIBS_INIT})

if(TARGET vulkan)
    list(APPEND definitions PRIVATE -DUNINSTALLED_LOADER="$<TARGET_FILE:vulkan>")
endif()

if(WIN32)
    list(APPEND definitions PRIVATE -DVK_USE_PLATFORM_WIN32_KHR)
    list(APPEND definitions PRIVATE -DWIN32_LEAN_AND_MEAN)

    list(APPEND sources ShellWin32.cpp ShellWin32.h)
else()
    list(APPEND libraries PRIVATE -ldl)

    find_package(XCB REQUIRED)

    list(APPEND sources ShellXcb.cpp ShellXcb.h)
    list(APPEND definitions PRIVATE -DVK_USE_PLATFORM_XCB_KHR)
    list(APPEND includes PRIVATE ${XCB_INCLUDES})
    list(APPEND libraries PRIVATE ${XCB_LIBRARIES})
endif()


add_executable(Hologram ${sources})
target_compile_definitions(Hologram ${definitions})
target_include_directories(Hologram ${includes})
target_link_libraries(Hologram ${libraries})
