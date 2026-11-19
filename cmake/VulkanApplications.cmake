# MaterialBaker resolves inputs relative to its own executable (AssetLoader walks
# up from the binary looking for an Asset/ directory), not its working directory.
# With a build tree outside the source tree it therefore cannot find
# Materials/splash.mymaterial or the textures it references, and dereferences a
# missing scene. Stage the source assets into the build tree first; this also
# makes scene, font, and texture assets discoverable by externally built
# Viewer/Billiard binaries, which use the same executable-relative search.
add_custom_target(VulkanApplicationMaterials
    COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
        "${PROJECT_SOURCE_DIR}/Asset/Scene" "${PROJECT_BINARY_DIR}/Asset/Scene"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
        "${PROJECT_SOURCE_DIR}/Asset/Fonts" "${PROJECT_BINARY_DIR}/Asset/Fonts"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
        "${PROJECT_SOURCE_DIR}/Asset/Textures" "${PROJECT_BINARY_DIR}/Asset/Textures"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
        "${PROJECT_SOURCE_DIR}/Asset/Materials" "${PROJECT_BINARY_DIR}/Asset/Materials"
    COMMAND "${CMAKE_COMMAND}" -E chdir "${PROJECT_BINARY_DIR}/Asset/Materials" "$<TARGET_FILE:MaterialBaker>"
    DEPENDS MaterialBaker
    VERBATIM)

function(myge_vulkan_application target)
    target_compile_definitions(${target} PRIVATE MYGE_USE_VULKAN)
    target_link_libraries(${target} BulletPhysics VulkanRHI Framework PlatformInterface)
    add_dependencies(${target} VulkanApplicationMaterials)
    if(APPLE)
        set(asset_root "$<TARGET_BUNDLE_DIR:${target}>/Contents/Resources/Asset")
        add_custom_target(${target}VulkanAssets
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different "${PROJECT_SOURCE_DIR}/Asset/Scene" "${asset_root}/Scene"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different "${PROJECT_SOURCE_DIR}/Asset/Fonts" "${asset_root}/Fonts"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different "${PROJECT_SOURCE_DIR}/Asset/Textures" "${asset_root}/Textures"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different "${PROJECT_BINARY_DIR}/Asset/Materials" "${asset_root}/Materials"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
                "${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan/Engine" "${asset_root}/Shaders/Vulkan/Engine"
            VERBATIM)
        add_dependencies(${target}VulkanAssets Engine_Asset VulkanApplicationMaterials)
        add_dependencies(${target} ${target}VulkanAssets)
    endif()
endfunction()

install(DIRECTORY "${PROJECT_SOURCE_DIR}/Asset/Scene" "${PROJECT_SOURCE_DIR}/Asset/Fonts"
    "${PROJECT_SOURCE_DIR}/Asset/Textures" DESTINATION Asset)
install(DIRECTORY "${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan/Engine" DESTINATION Asset/Shaders/Vulkan)
install(DIRECTORY "${PROJECT_BINARY_DIR}/Asset/Materials" DESTINATION Asset)
