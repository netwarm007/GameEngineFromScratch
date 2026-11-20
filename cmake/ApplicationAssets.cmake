include_guard(GLOBAL)

# AssetLoader searches relative to the executable, not MaterialBaker's cwd.
add_custom_target(EngineApplicationMaterials
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
