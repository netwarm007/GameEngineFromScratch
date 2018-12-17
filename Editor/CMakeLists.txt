IF(APPLE)
    set(SHADER_SOURCES basic.vert basic.frag
                debug.vert debug.frag
                pbr.vert pbr.frag
                skybox.vert skybox.frag
                shadowmap.vert shadowmap.frag
                integrateBRDF.comp
            )

    foreach(SHADER IN LISTS SHADER_SOURCES)
        add_custom_command(OUTPUT ${SHADER}.air
            COMMAND xcrun -sdk macosx metal -g -MO -c ${PROJECT_SOURCE_DIR}/Asset/Shaders/Metal/${SHADER}.metal -o ${SHADER}.air
            COMMENT "Compile ${SHADER}.metal --> ${SHADER}.air"
            DEPENDS ${PROJECT_SOURCE_DIR}/Asset/Shaders/Metal/${SHADER}.metal
            )

        list(APPEND AIRS ${SHADER}.air)
    endforeach(SHADER)

    add_custom_command(OUTPUT Editor.metalar
            COMMAND xcrun -sdk macosx metal-ar rcv Editor.metalar ${AIRS}
            COMMENT "Archive ${AIRS} --> Editor.metalar"
            DEPENDS ${AIRS}
        )

    add_custom_command(OUTPUT Editor.metallib
            COMMAND xcrun -sdk macosx metallib Editor.metalar -split-module-without-linking -o Editor.metallib
            COMMENT "Compile Editor.metalar --> Editor.metallib"
            DEPENDS Editor.metalar
        )

    add_custom_target(EditorShaderLibrary
                    DEPENDS Editor.metallib
        )

    add_executable(Editor MACOSX_BUNDLE
        EditorLogic.cpp 
        EditorConfig.cpp
        Editor.metallib
        )

    add_dependencies(Editor EditorShaderLibrary)

    set(RESOURCE_FILES
        Editor.metallib
    )

    set_target_properties(Editor PROPERTIES
        RESOURCE "${RESOURCE_FILES}"
        )
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
  IF(APPLE)
    target_link_libraries(Editor
                            MyPhysics
                            MyGameEngineMetal
                        )
  ELSE(APPLE)
    target_link_libraries(Editor
                            MyPhysics
                            MyGameEngineOpenGL
                        )
  ENDIF(APPLE)
ENDIF(WIN32)
