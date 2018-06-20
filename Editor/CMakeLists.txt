IF(APPLE)
add_executable(Editor MACOSX_BUNDLE
    EditorLogic.cpp EditorConfig.cpp)
ELSE(APPLE)
add_executable(Editor
    EditorLogic.cpp EditorConfig.cpp)
ENDIF(APPLE)
IF(WIN32)
    target_link_libraries(Editor
                            MyPhysics
                            MyGameEngineD3d12
                        )
    add_custom_command(TARGET Editor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
            $<TARGET_FILE_DIR:Editor>)
ELSE(WIN32)
    target_link_libraries(Editor
                            MyPhysics
                            MyGameEngine
                        )
ENDIF(WIN32)
