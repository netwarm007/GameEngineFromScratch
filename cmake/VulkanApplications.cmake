function(myge_vulkan_application target)
    target_compile_definitions(${target} PRIVATE MYGE_USE_VULKAN)
    target_link_libraries(${target} BulletPhysics VulkanRHI Framework PlatformInterface)
    if(APPLE)
        set(asset_root "$<TARGET_BUNDLE_DIR:${target}>/Contents/Resources/Asset")
        add_custom_target(${target}VulkanAssets
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different "${PROJECT_SOURCE_DIR}/Asset/Scene" "${asset_root}/Scene"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different "${PROJECT_SOURCE_DIR}/Asset/Fonts" "${asset_root}/Fonts"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different "${PROJECT_SOURCE_DIR}/Asset/Textures" "${asset_root}/Textures"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
                "${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan/Engine" "${asset_root}/Shaders/Vulkan/Engine"
            VERBATIM)
        add_dependencies(${target}VulkanAssets Engine_Asset)
        add_dependencies(${target} ${target}VulkanAssets)
    endif()
endfunction()

install(DIRECTORY "${PROJECT_SOURCE_DIR}/Asset/Scene" "${PROJECT_SOURCE_DIR}/Asset/Fonts"
    "${PROJECT_SOURCE_DIR}/Asset/Textures" DESTINATION Asset)
install(DIRECTORY "${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan/Engine" DESTINATION Asset/Shaders/Vulkan)
