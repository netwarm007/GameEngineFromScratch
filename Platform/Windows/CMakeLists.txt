add_library(WindowsApplication WindowsApplication.cpp)

add_executable(MyGameEngineD3d12 
    D3d12Application.cpp
    D3d12ApplicationConfig.cpp)
target_link_libraries(MyGameEngineD3d12 WindowsApplication Common ${OPENGEX_LIBRARY} ${OPENDDL_LIBRARY} ${XG_LIBRARY} ${ZLIB_LIBRARY} D3dRHI d3d12 dxgi)
__add_xg_platform_dependencies(MyGameEngineD3d12)

add_custom_command(TARGET MyGameEngineD3d12 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
        $<TARGET_FILE_DIR:MyGameEngineD3d12>)

find_package(OpenGL REQUIRED)

add_executable(MyGameEngineOpenGL 
    OpenGLApplication.cpp
    OpenGLApplicationConfig.cpp
    ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_wgl.c
    )
target_link_libraries(MyGameEngineOpenGL WindowsApplication Common  ${OPENGEX_LIBRARY} ${OPENDDL_LIBRARY} ${XG_LIBRARY} ${ZLIB_LIBRARY} OpenGLRHI ${OPENGL_LIBRARIES})
target_include_directories(MyGameEngineOpenGL PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
__add_xg_platform_dependencies(MyGameEngineOpenGL)

add_custom_command(TARGET MyGameEngineOpenGL POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
        $<TARGET_FILE_DIR:MyGameEngineOpenGL>)

add_subdirectory(test)

