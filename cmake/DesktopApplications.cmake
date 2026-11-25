include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/ApplicationAssets.cmake")

set(MYGE_DESKTOP_RHI_TARGETS VulkanRHI)
set(MYGE_DESKTOP_SHADER_DIRECTORIES Vulkan/Engine)
if(WIN32)
    list(APPEND MYGE_DESKTOP_RHI_TARGETS D3dRHI OpenGLRHI)
    list(APPEND MYGE_DESKTOP_SHADER_DIRECTORIES HLSL OpenGL)
elseif(APPLE)
    list(APPEND MYGE_DESKTOP_RHI_TARGETS MetalRHI)
else()
    list(APPEND MYGE_DESKTOP_RHI_TARGETS OpenGLRHI)
    list(APPEND MYGE_DESKTOP_SHADER_DIRECTORIES OpenGL)
endif()

function(myge_desktop_application target)
    target_link_libraries(${target} ${MYGE_DESKTOP_RHI_TARGETS})
    add_dependencies(${target} Engine_Asset EngineApplicationMaterials)

    if(APPLE)
        set(asset_root "$<TARGET_BUNDLE_DIR:${target}>/Contents/Resources/Asset")
        add_custom_target(${target}DesktopAssets
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
                "${PROJECT_BINARY_DIR}/Asset/Scene" "${asset_root}/Scene"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
                "${PROJECT_BINARY_DIR}/Asset/Fonts" "${asset_root}/Fonts"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
                "${PROJECT_BINARY_DIR}/Asset/Textures" "${asset_root}/Textures"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
                "${PROJECT_BINARY_DIR}/Asset/Materials" "${asset_root}/Materials"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
                "${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan/Engine" "${asset_root}/Shaders/Vulkan/Engine"
            DEPENDS Engine_Asset EngineApplicationMaterials
            VERBATIM)
        add_dependencies(${target} ${target}DesktopAssets)
    endif()
endfunction()

install(DIRECTORY "${PROJECT_BINARY_DIR}/Asset/Scene"
    "${PROJECT_BINARY_DIR}/Asset/Fonts" "${PROJECT_BINARY_DIR}/Asset/Textures"
    "${PROJECT_BINARY_DIR}/Asset/Materials" DESTINATION Asset)
foreach(shader_directory IN LISTS MYGE_DESKTOP_SHADER_DIRECTORIES)
    get_filename_component(shader_parent "${shader_directory}" DIRECTORY)
    install(DIRECTORY "${PROJECT_BINARY_DIR}/Asset/Shaders/${shader_directory}"
        DESTINATION "Asset/Shaders/${shader_parent}")
endforeach()
