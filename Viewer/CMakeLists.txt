IF(ANDROID)
    # Export ANativeActivity_onCreate(),
    # Refer to: https://github.com/android-ndk/ndk/issues/381.
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

    add_library(Viewer SHARED 
        ViewerLogic.cpp ViewerConfig.cpp)
ELSE(ANDROID)
    add_executable(Viewer MACOSX_BUNDLE
        ViewerLogic.cpp ViewerConfig.cpp)
ENDIF(ANDROID)

IF(APPLE)
    target_link_libraries(Viewer
                            MyPhysics
                            Common
                            MetalRHI
                            PlatformInterface
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
            PROPERTIES LINK_FLAGS "-s FULL_ES2=1 -s FULL_ES3=1 -s USE_WEBGL2=1 -s USE_SDL=2 --shell-file ${CMAKE_CURRENT_SOURCE_DIR}/Viewer.html"
            )
ELSE(APPLE)
    target_link_libraries(Viewer
                            MyPhysics
                            OpenGLRHI
                            Common
                            PlatformInterface
                        )
ENDIF(APPLE)

IF(WIN32)
    add_custom_command(TARGET Viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
            $<TARGET_FILE_DIR:Viewer>)
ENDIF(WIN32)

target_include_directories(BilliardGame PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
