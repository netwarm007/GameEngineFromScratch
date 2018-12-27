IF(APPLE)
    find_library(CEF_LIBRARY NAMES "Chromium Embedded Framework" PATHS ${MYGE_EXTERNAL_FRAMEWORK_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
    find_library(CEF_WRAPPER NAMES cef_dll_wrapper libcef_dll_wrapper PATHS ${MYGE_EXTERNAL_LIBRARY_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)

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

    # main app
    add_executable(Editor MACOSX_BUNDLE
        EditorLogic.cpp 
        EditorConfig.cpp
        Editor.metallib
        )

    set_target_properties(Editor PROPERTIES
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/mac/Info.plist
        )

    add_dependencies(Editor EditorShaderLibrary)

    set(RESOURCE_FILES
        Editor.metallib
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
        ${CEF_WRAPPER}
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
                        MyGameEngineCef
                    )

IF(WIN32)
    set(OTHER_DEPENDENCIES
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/libcef.dll"
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/chrome_elf.dll"
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/libEGL.dll"
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/libGLESv2.dll"
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/natives_blob.bin"
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

        # Copy the Helper app into the Frameworks directory.
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CEF_HELPER_OUTPUT_NAME}.app
                $<TARGET_FILE_DIR:Editor>/../Frameworks/${CEF_HELPER_OUTPUT_NAME}.app

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
            "${PROJECT_SOURCE_DIR}/External/Linux/bin/natives_blob.bin"
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
