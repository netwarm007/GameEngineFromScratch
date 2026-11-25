cmake_minimum_required(VERSION 4.0)
project(DesktopApplicationFixture LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    set(expected_backends VulkanRHI D3dRHI OpenGLRHI)
    set(shader_files Vulkan/Engine/pbr.frag.spv HLSL/pbr.frag.cso OpenGL/pbr.frag.glsl)
elseif(APPLE)
    set(expected_backends VulkanRHI MetalRHI)
    set(shader_files Vulkan/Engine/pbr.frag.spv)
else()
    set(expected_backends VulkanRHI OpenGLRHI)
    set(shader_files Vulkan/Engine/pbr.frag.spv OpenGL/pbr.frag.glsl)
endif()

set(declarations "")
set(calls "")
foreach(backend IN LISTS expected_backends)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${backend}.cpp" "void ${backend}() {}\n")
    add_library(${backend} STATIC "${CMAKE_CURRENT_BINARY_DIR}/${backend}.cpp")
    string(APPEND declarations "void ${backend}();\n")
    string(APPEND calls "${backend}();\n")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp"
    "${declarations}int main() {\n${calls}return 0;\n}\n")

foreach(asset Scene/splash.ogex Fonts/font.ttf Textures/icon.png Materials/splash.mymaterial)
    get_filename_component(parent "${asset}" DIRECTORY)
    file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/Asset/${parent}")
    file(WRITE "${PROJECT_SOURCE_DIR}/Asset/${asset}" "fixture:${asset}\n")
endforeach()
foreach(shader IN LISTS shader_files)
    get_filename_component(parent "${shader}" DIRECTORY)
    file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/Asset/Shaders/${parent}")
    file(WRITE "${PROJECT_SOURCE_DIR}/Asset/Shaders/${shader}" "fixture:${shader}\n")
endforeach()
add_custom_target(Engine_Asset
    COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
        "${PROJECT_SOURCE_DIR}/Asset/Shaders" "${PROJECT_BINARY_DIR}/Asset/Shaders"
    VERBATIM)

# Substitute only the baker and GPU libraries; exercise the production build
# graph, executable-relative input staging, and bundle/install copy commands.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/MaterialBaker.cpp" [=[
#include <filesystem>
#include <fstream>
int main(int argc, char** argv) {
    if (argc == 0) return 1;
    auto root = std::filesystem::path(argv[0]).parent_path().parent_path();
    if (!std::filesystem::exists(root / "Asset/Materials/splash.mymaterial") ||
        !std::filesystem::exists(root / "Asset/Textures/icon.png")) return 2;
    std::ofstream output("baked.pvr");
    output << "baked fixture\n";
    return output ? 0 : 3;
}
]=])
add_executable(MaterialBaker "${CMAKE_CURRENT_BINARY_DIR}/MaterialBaker.cpp")
set_target_properties(MaterialBaker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/Utility/$<0:>")

include("${ENGINE_SOURCE_DIR}/cmake/DesktopApplications.cmake")
# Do not let screenshot discovery mistake these fixtures for real applications.
foreach(app FixtureViewer FixtureBilliard)
    add_executable(${app} MACOSX_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")
    if(APPLE)
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${app}.metallib" "fixture:metallib\n")
        target_sources(${app} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/${app}.metallib")
        set_target_properties(${app} PROPERTIES
            RESOURCE "${CMAKE_CURRENT_BINARY_DIR}/${app}.metallib")
    endif()
    myge_desktop_application(${app})
    install(TARGETS ${app} RUNTIME BUNDLE DESTINATION bin)
endforeach()
