include("${CMAKE_CURRENT_LIST_DIR}/ApplicationAssets.cmake")

function(myge_vulkan_application target)
    target_compile_definitions(${target} PRIVATE MYGE_USE_VULKAN)
    target_link_libraries(${target} BulletPhysics VulkanRHI Framework PlatformInterface)
    add_dependencies(${target} EngineApplicationMaterials)
    if(BUILD_ALL_DESKTOP_RHIS)
        return()
    endif()
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
        add_dependencies(${target}VulkanAssets Engine_Asset EngineApplicationMaterials)
        add_dependencies(${target} ${target}VulkanAssets)
    endif()
endfunction()

if(NOT BUILD_ALL_DESKTOP_RHIS)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/Asset/Scene" "${PROJECT_SOURCE_DIR}/Asset/Fonts"
        "${PROJECT_SOURCE_DIR}/Asset/Textures" DESTINATION Asset)
    install(DIRECTORY "${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan/Engine" DESTINATION Asset/Shaders/Vulkan)
    install(DIRECTORY "${PROJECT_BINARY_DIR}/Asset/Materials" DESTINATION Asset)
endif()
