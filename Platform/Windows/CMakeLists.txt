add_executable(MyGameEngine WindowsApplication.cpp D3d12GraphicsManager.cpp)
target_link_libraries(MyGameEngine Common d3d12 dxgi d3dcompiler)
add_custom_command(TARGET MyGameEngine PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/Asset/Shaders $<TARGET_FILE_DIR:MyGameEngine>)

