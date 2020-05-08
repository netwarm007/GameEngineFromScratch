IF(APPLE)
    set(SHADER_SOURCES basic.vert basic.frag
                debug.vert debug.frag
                pbr.vert pbr.frag
                skybox.vert skybox.frag
                shadowmap.vert shadowmap.frag
                shadowmap_omni.vert shadowmap_omni.frag
                terrain.vert terrain.frag
                texture.frag texturearray.frag
                passthrough.vert passthrough_cube.vert
                integrateBRDF.comp
            )

    foreach(SHADER IN LISTS SHADER_SOURCES)
        add_custom_command(OUTPUT ${SHADER}.air
            COMMAND xcrun -sdk macosx metal -g -MO -c ${PROJECT_BINARY_DIR}/Asset/Shaders/Metal/${SHADER}.metal -o ${SHADER}.air
            COMMENT "Compile ${SHADER}.metal --> ${SHADER}.air"
            DEPENDS Engine_Asset_Shaders
            )

        list(APPEND AIRS ${SHADER}.air)
    endforeach(SHADER)

    add_custom_command(OUTPUT Main.metalar
            COMMAND xcrun -sdk macosx metal-ar rcv Main.metalar ${AIRS}
            COMMENT "Archive ${AIRS} --> Main.metalar"
            DEPENDS ${AIRS}
        )

    add_custom_command(OUTPUT Main.metallib
            COMMAND xcrun -sdk macosx metallib Main.metalar -o Main.metallib
            COMMENT "Compile Main.metalar --> Main.metallib"
            DEPENDS Main.metalar
        )

    add_custom_target(EditorShaderLibrary
                    DEPENDS Main.metallib
        )

    # main app
    add_executable(Editor MACOSX_BUNDLE
        EditorLogic.cpp 
        EditorConfig.cpp
        Main.metallib
        )

    set_target_properties(Editor PROPERTIES
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/mac/Info.plist
        )

    add_dependencies(Editor EditorShaderLibrary)

    set(RESOURCE_FILES
        Main.metallib
    )

    set_target_properties(Editor PROPERTIES
        RESOURCE "${RESOURCE_FILES}"
    )

    # helper app
    set(CEF_HELPER_OUTPUT_NAME "Editor Helper")
    add_executable(EditorHelper MACOSX_BUNDLE
        process_helper_mac.cpp
    )

    target_link_libraries(EditorHelper
        CEF
        PlatformInterface
    )

    set_target_properties(EditorHelper PROPERTIES
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/mac/helper-Info.plist
        OUTPUT_NAME ${CEF_HELPER_OUTPUT_NAME}
        )
ELSE(APPLE)
    IF(WIN32)
        add_executable(Editor
            EditorLogic.cpp EditorConfig.cpp win/Editor.exe.manifest win/Editor.rc)
    ELSE(WIN32)
        add_executable(Editor
            EditorLogic.cpp EditorConfig.cpp)
    ENDIF(WIN32)
ENDIF(APPLE)

target_link_libraries(Editor
                        MyPhysics
                        EmptyRHI
                        Common
                        CEF
                        PlatformInterface
                    )

# copy dependencies
IF(WIN32)
    set(OTHER_DEPENDENCIES
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/libcef.dll"
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/chrome_elf.dll"
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/libEGL.dll"
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/libGLESv2.dll"
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/snapshot_blob.bin"
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/v8_context_snapshot.bin"
    )
    add_custom_command(TARGET Editor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OTHER_DEPENDENCIES}
            $<TARGET_FILE_DIR:Editor>
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/win/Resources
            $<TARGET_FILE_DIR:Editor>
        COMMAND "mt.exe" -nologo
            -manifest ${CMAKE_CURRENT_SOURCE_DIR}/win/Editor.exe.manifest ${CMAKE_CURRENT_SOURCE_DIR}/win/compatibility.manifest
            -outputresource:$<TARGET_FILE:Editor>\;\#1
    )
ELSE(WIN32)
  IF(APPLE)
    # Copy files into the main app bundle.
    add_custom_command(
        TARGET Editor
        POST_BUILD
        # Copy the CEF framework into the Frameworks directory.
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${PROJECT_SOURCE_DIR}/External/Darwin/framework
                $<TARGET_FILE_DIR:Editor>/../Frameworks

        # Move the Helper app into the Frameworks directory and change the rpath.
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                $<TARGET_BUNDLE_DIR:EditorHelper>
                $<TARGET_FILE_DIR:Editor>/../Frameworks/${CEF_HELPER_OUTPUT_NAME}.app
        
        COMMAND install_name_tool -change 
                "@executable_path/../Frameworks/Chromium Embedded Framework.framework/Chromium Embedded Framework" 
                "@executable_path/../../../Chromium Embedded Framework.framework/Chromium Embedded Framework"
                $<TARGET_FILE_DIR:Editor>/../Frameworks/${CEF_HELPER_OUTPUT_NAME}.app/Contents/MacOS/${CEF_HELPER_OUTPUT_NAME}

        # Copy the resources
        COMMAND ${CMAKE_COMMAND} -E make_directory
                $<TARGET_FILE_DIR:Editor>/../Resources/English.lproj
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/mac/English.lproj/InfoPlist.strings
                $<TARGET_FILE_DIR:Editor>/../Resources/English.lproj/
        COMMAND /usr/bin/ibtool --output-format binary1 --compile 
                $<TARGET_FILE_DIR:Editor>/../Resources/English.lproj/MainMenu.nib
                ${CMAKE_CURRENT_SOURCE_DIR}/mac/English.lproj/MainMenu.xib
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/mac/Editor.icns
                $<TARGET_FILE_DIR:Editor>/../Resources/

        VERBATIM
    )
  ELSE(APPLE)
    set(OTHER_DEPENDENCIES
            "${PROJECT_SOURCE_DIR}/External/Linux/lib/libcef.so"
            "${PROJECT_SOURCE_DIR}/External/Linux/lib/libEGL.so"
            "${PROJECT_SOURCE_DIR}/External/Linux/lib/libGLESv2.so"
            "${PROJECT_SOURCE_DIR}/External/Linux/bin/snapshot_blob.bin"
            "${PROJECT_SOURCE_DIR}/External/Linux/bin/v8_context_snapshot.bin"
    )
    add_custom_command(TARGET Editor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OTHER_DEPENDENCIES}
            $<TARGET_FILE_DIR:Editor>
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/nix/Resources
            $<TARGET_FILE_DIR:Editor>
    )
  ENDIF(APPLE)
ENDIF(WIN32)
