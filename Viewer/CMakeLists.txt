IF(ANDROID)
    # Export ANativeActivity_onCreate(),
    # Refer to: https://github.com/android-ndk/ndk/issues/381.
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

    add_library(Viewer SHARED 
        ViewerLogic.cpp ViewerConfig.cpp)
ELSEIF(APPLE)
    add_executable(Viewer MACOSX_BUNDLE
        ViewerLogic.cpp ViewerConfig.cpp
        Main.metallib
        )
ELSE(ANDROID)
    add_executable(Viewer 
        ViewerLogic.cpp ViewerConfig.cpp
        )
ENDIF(ANDROID)

set(SHADER_SOURCES basic.vert basic.frag
            debug.vert debug.frag
            pbr.vert pbr.frag
            skybox.vert skybox.frag
            shadowmap.vert shadowmap.frag
            shadowmap_omni.vert shadowmap_omni.frag
            terrain.vert terrain.frag
            texture.frag texturearray.frag
            cubemap.frag cubemaparray.frag
            passthrough.vert passthrough_cube.vert
            integrateBRDF.comp
            raytrace.comp
        )

IF(APPLE)
    foreach(SHADER IN LISTS SHADER_SOURCES)
        add_custom_command(OUTPUT ${SHADER}.air
            COMMAND xcrun -sdk macosx metal -g -MO -c ${PROJECT_BINARY_DIR}/Asset/Shaders/Metal/${SHADER}.metal -o ${SHADER}.air
            COMMENT "Compile ${SHADER}.metal --> ${SHADER}.air"
            DEPENDS Engine_Asset_Shaders
            )

        list(APPEND AIRS ${SHADER}.air)
    endforeach(SHADER)

    add_custom_command(OUTPUT Main.metalar
            COMMAND xcrun -sdk macosx metal-ar rucv Main.metalar ${AIRS}
            COMMENT "Archive ${AIRS} --> Main.metalar"
            DEPENDS ${AIRS}
        )

    add_custom_command(OUTPUT Main.metallib
            COMMAND xcrun -sdk macosx metallib Main.metalar -o Main.metallib
            COMMENT "Compile Main.metalar --> Main.metallib"
            DEPENDS Main.metalar
        )

    add_custom_target(ViewerShaderLibrary
                    DEPENDS Main.metallib
        )

    target_link_libraries(Viewer
                            MyPhysics
                            Common
                            MetalRHI
                            PlatformInterface
                        )

    set_target_properties(Viewer PROPERTIES
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/mac/Info.plist
        )

    add_dependencies(Viewer ViewerShaderLibrary)

    set(RESOURCE_FILES
        Main.metallib
    )

    set_target_properties(Viewer PROPERTIES
        RESOURCE "${RESOURCE_FILES}"
    )

ELSEIF(WA)
    target_link_libraries(Viewer
                            MyPhysics
                            OpenGLRHI
                            Common
                            Sdl
                        )

    add_custom_target(SplashAssets ALL
        python $ENV{EMSCRIPTEN}/tools/file_packager.py ${CMAKE_CURRENT_BINARY_DIR}/Asset.data 
            --lz4
            --no-heap-copy
            --js-output=${CMAKE_CURRENT_BINARY_DIR}/Asset.preload.js 
            --preload Asset/Textures/gold-scuffed_basecolor.png
                    Asset/Textures/gold-scuffed_normal.png
                    Asset/Textures/gold-scuffed_metallic.png
                    Asset/Textures/gold-scuffed_roughness.png
                    Asset/Textures/default_ao.png
                    Asset/Textures/redbricks2b-albedo.png
                    Asset/Textures/redbricks2b-normal.png
                    Asset/Textures/redbricks2b-metalness.png
                    Asset/Textures/redbricks2b-rough.png
                    Asset/Textures/redbricks2b-ao.png
                    Asset/Textures/terrain/area_1_0_0.png
                    Asset/Textures/hdr/spruit_sunrise_posx.dds
                    Asset/Textures/hdr/spruit_sunrise_posy.dds
                    Asset/Textures/hdr/spruit_sunrise_posz.dds
                    Asset/Textures/hdr/spruit_sunrise_negx.dds
                    Asset/Textures/hdr/spruit_sunrise_negy.dds
                    Asset/Textures/hdr/spruit_sunrise_negz.dds
                    Asset/Textures/hdr/spruit_sunrise_irradiance_posx.dds
                    Asset/Textures/hdr/spruit_sunrise_irradiance_posy.dds
                    Asset/Textures/hdr/spruit_sunrise_irradiance_posz.dds
                    Asset/Textures/hdr/spruit_sunrise_irradiance_negx.dds
                    Asset/Textures/hdr/spruit_sunrise_irradiance_negy.dds
                    Asset/Textures/hdr/spruit_sunrise_irradiance_negz.dds
                    Asset/Textures/hdr/spruit_sunrise_radiance_posx.dds
                    Asset/Textures/hdr/spruit_sunrise_radiance_posy.dds
                    Asset/Textures/hdr/spruit_sunrise_radiance_posz.dds
                    Asset/Textures/hdr/spruit_sunrise_radiance_negx.dds
                    Asset/Textures/hdr/spruit_sunrise_radiance_negy.dds
                    Asset/Textures/hdr/spruit_sunrise_radiance_negz.dds
                    Asset/Scene/splash.ogex
                    Asset/Shaders/WebGL
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        VERBATIM
            )

    set_target_properties(Viewer
            PROPERTIES LINK_FLAGS "-s FULL_ES2=1 -s FULL_ES3=1 -s USE_WEBGL2=1 -s USE_SDL=2 -s DEMANGLE_SUPPORT=1 --shell-file ${CMAKE_CURRENT_SOURCE_DIR}/Viewer.html"
            )
ELSEIF(WIN32)
    target_link_libraries(Viewer
                            MyPhysics
                            OpenGLRHI 
                            Common
                            PlatformInterface
                        )
ELSE(APPLE)
    target_link_libraries(Viewer
                            MyPhysics
                            OpenGLRHI
                            Common
                            PlatformInterface
                        )
ENDIF(APPLE)

add_dependencies(Viewer Engine_Asset)
