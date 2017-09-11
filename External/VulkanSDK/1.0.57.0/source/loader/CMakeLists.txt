include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_PROJECT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}
)

# Check for the existance of the secure_getenv or __secure_getenv commands
include(CheckFunctionExists)
CHECK_FUNCTION_EXISTS(secure_getenv HAVE_SECURE_GETENV)
CHECK_FUNCTION_EXISTS(__secure_getenv HAVE___SECURE_GETENV)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/loader_cmake_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/loader_cmake_config.h)

# Fetch header version from vulkan.h
file(STRINGS "../include/vulkan/vulkan.h" lines REGEX "^#define VK_HEADER_VERSION [0-9]+")
list(LENGTH lines len)
if(${len} EQUAL 1)
    string(REGEX MATCHALL "[0-9]+" vk_header_version ${lines})
else()
    MESSAGE(FATAL_ERROR "Unable to fetch version from vulkan.h")
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
       add_definitions(-DVK_USE_PLATFORM_XLIB_KHR -DVK_USE_PLATFORM_XLIB_XRANDR_EXT)
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

if (WIN32)
	FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-${MAJOR}.def DEF_FILE)
	FILE(READ ${DEF_FILE} DEF_TEXT)
	STRING(REGEX REPLACE "vulkan-${MAJOR}.dll" "${API_LOWERCASE}-${MAJOR}.dll" DEF_EDITED "${DEF_TEXT}" )
	FILE(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${API_LOWERCASE}-${MAJOR}.def" "${DEF_EDITED}")
endif()

# DEBUG enables runtime loader ICD verification
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

set(NORMAL_LOADER_SRCS
    extension_manual.c
    loader.c
    loader.h
    vk_loader_platform.h
    vk_loader_layer.h
    trampoline.c
    wsi.c
    wsi.h
    debug_report.c
    debug_report.h
    gpa_helper.h
    cJSON.c
    cJSON.h
    murmurhash.c
    murmurhash.h
)

set(OPT_LOADER_SRCS
    dev_ext_trampoline.c
    phys_dev_ext.c
)

add_definitions(-DAPI_NAME="${API_NAME}")

# Check for assembler support
set(ASM_FAILURE_MSG "The build will fall back on building with C code\n")
set(ASM_FAILURE_MSG "${ASM_FAILURE_MSG}Note that this may be unsafe, as the C code requires tail-call optimizations to remove")
set(ASM_FAILURE_MSG "${ASM_FAILURE_MSG} the stack frame for certain calls. If the compiler does not do this, then unknown device")
set(ASM_FAILURE_MSG "${ASM_FAILURE_MSG} extensions will suffer from a corrupted stack.")
if (WIN32)
    enable_language(ASM_MASM)
    if (CMAKE_ASM_MASM_COMPILER_WORKS)
        if (NOT CMAKE_CL_64)
            set(CMAKE_ASM_MASM_FLAGS ${CMAKE_ASM_MASM_FLAGS} /safeseh)
        endif()
        set(OPT_LOADER_SRCS ${OPT_LOADER_SRCS} unknown_ext_chain_masm.asm)

        add_executable(asm_offset asm_offset.c)
        add_dependencies(asm_offset generate_helper_files loader_gen_files)
        add_custom_command(OUTPUT gen_defines.asm DEPENDS asm_offset COMMAND asm_offset MASM)
        add_custom_target(loader_asm_gen_files DEPENDS gen_defines.asm)
    else()
        message(WARNING "Could not find working MASM assebler\n${ASM_FAILURE_MSG}")
        set(OPT_LOADER_SRCS ${OPT_LOADER_SRCS} unknown_ext_chain.c)
        add_custom_target(loader_asm_gen_files)
    endif()
else()
    enable_language(ASM-ATT)
    set(CMAKE_ASM-ATT_FLAGS "${CMAKE_ASM-ATT_FLAGS} $ENV{ASFLAGS}")
    set(CMAKE_ASM-ATT_FLAGS "${CMAKE_ASM-ATT_FLAGS} -I\"${CMAKE_CURRENT_BINARY_DIR}\"")

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/asm_test.asm ".intel_syntax noprefix\n.text\n.global sample\nsample:\nmov ecx, [eax + 16]\n")
    try_compile(ASSEMBLER_WORKS ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/asm_test.asm)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/asm_test.asm)
    if (ASSEMBLER_WORKS)
        set(OPT_LOADER_SRCS ${OPT_LOADER_SRCS} unknown_ext_chain_gas.asm)
        add_executable(asm_offset asm_offset.c)
        add_dependencies(asm_offset generate_helper_files loader_gen_files)
        add_custom_command(OUTPUT gen_defines.asm DEPENDS asm_offset COMMAND asm_offset GAS)
        add_custom_target(loader_asm_gen_files DEPENDS gen_defines.asm)
    else()
        message(WARNING "Could not find working x86 GAS assembler\n${ASM_FAILURE_MSG}")
        set(OPT_LOADER_SRCS ${OPT_LOADER_SRCS} unknown_ext_chain.c)
        add_custom_target(loader_asm_gen_files)
    endif()
endif()

run_vk_xml_generate(loader_extension_generator.py vk_loader_extensions.h)
run_vk_xml_generate(loader_extension_generator.py vk_loader_extensions.c)
add_custom_target(loader_gen_files DEPENDS
        vk_loader_extensions.h
        vk_loader_extensions.c
    )

if (WIN32)
    # Use static MSVCRT libraries
    foreach(configuration in CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_RELWITHDEBINFO
                             CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${configuration} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${configuration} "${${configuration}}")
        endif()
    endforeach()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_CRT_SECURE_NO_WARNINGS")
    # Build dev_ext_trampoline.c with -O2 to allow tail-call optimization.
    # Build other C files with normal options
    # setup two Cmake targets (loader-norm and loader-opt) for the different compilation flags
    separate_arguments(LOCAL_C_FLAGS_DBG WINDOWS_COMMAND ${CMAKE_C_FLAGS_DEBUG})
    set(CMAKE_C_FLAGS_DEBUG " ")
    separate_arguments(LOCAL_C_FLAGS_REL WINDOWS_COMMAND ${CMAKE_C_FLAGS_RELEASE})

    add_library(loader-norm OBJECT ${NORMAL_LOADER_SRCS} dirent_on_windows.c)
    add_dependencies(loader-norm generate_helper_files loader_gen_files)

    target_compile_options(loader-norm PUBLIC "$<$<CONFIG:DEBUG>:${LOCAL_C_FLAGS_DBG}>")
    add_library(loader-opt OBJECT ${OPT_LOADER_SRCS})
    add_dependencies(loader-opt generate_helper_files loader_gen_files loader_asm_gen_files)

    target_compile_options(loader-opt PUBLIC "$<$<CONFIG:DEBUG>:${LOCAL_C_FLAGS_REL}>")
    add_library(${API_LOWERCASE}-${MAJOR} SHARED $<TARGET_OBJECTS:loader-opt> $<TARGET_OBJECTS:loader-norm> ${CMAKE_CURRENT_BINARY_DIR}/${API_LOWERCASE}-${MAJOR}.def ${CMAKE_CURRENT_SOURCE_DIR}/loader.rc)
    add_library(VKstatic.${MAJOR} STATIC $<TARGET_OBJECTS:loader-opt> $<TARGET_OBJECTS:loader-norm>)
    # Suppress conflicting libs warning for debug builds.
    set_target_properties(${API_LOWERCASE}-${MAJOR} PROPERTIES LINK_FLAGS_DEBUG /ignore:4098)
    set_target_properties(VKstatic.${MAJOR} PROPERTIES OUTPUT_NAME VKstatic.${MAJOR})
    target_link_libraries(${API_LOWERCASE}-${MAJOR} shlwapi Cfgmgr32)
    add_dependencies(${API_LOWERCASE}-${MAJOR} generate_helper_files loader_gen_files loader_asm_gen_files)

    target_link_libraries(VKstatic.${MAJOR} shlwapi)
    if (CMAKE_GENERATOR MATCHES "^Visual Studio.*")
        file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>/${API_LOWERCASE}-${MAJOR}.dll COPY_SRC_PATH)
        file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR}/../demos/$<CONFIGURATION>/ COPY_DST_PATH)
        file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR}/../tests/$<CONFIGURATION>/ COPY_DST_TEST_PATH)
    else()
        file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR}/${API_LOWERCASE}-${MAJOR}.dll COPY_SRC_PATH)
        file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR}/../demos/ COPY_DST_PATH)
        file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR}/../tests/ COPY_DST_TEST_PATH)
    endif()
    add_custom_command(TARGET ${API_LOWERCASE}-${MAJOR} POST_BUILD
      COMMAND xcopy /Y /I ${COPY_SRC_PATH} ${COPY_DST_PATH})
    add_custom_command(TARGET ${API_LOWERCASE}-${MAJOR} POST_BUILD
      COMMAND xcopy /Y /I ${COPY_SRC_PATH} ${COPY_DST_TEST_PATH})

else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpointer-arith")

    add_library(${API_LOWERCASE} SHARED ${NORMAL_LOADER_SRCS} ${OPT_LOADER_SRCS})
    add_dependencies(${API_LOWERCASE} generate_helper_files loader_gen_files loader_asm_gen_files)
    set_target_properties(${API_LOWERCASE} PROPERTIES SOVERSION "1" VERSION "1.0.${vk_header_version}")
    target_link_libraries(${API_LOWERCASE} -ldl -lpthread -lm)

    if(INSTALL_LVL_FILES)
        install(TARGETS ${API_LOWERCASE} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()

    # Generate pkg-config file.
    include(FindPkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        set(VK_API_VERSION "1.0.${vk_header_version}")
        foreach(LIB ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES} ${PLATFORM_LIBS})
            set(PRIVATE_LIBS "${PRIVATE_LIBS} -l${LIB}")
        endforeach()
        configure_file("vulkan.pc.in" "vulkan.pc" @ONLY)
        if(INSTALL_LVL_FILES)
            install(FILES       "${CMAKE_CURRENT_BINARY_DIR}/vulkan.pc"
                    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
        endif()
    endif()
endif()
