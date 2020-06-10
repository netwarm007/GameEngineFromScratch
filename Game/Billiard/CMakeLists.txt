IF(ANDROID)
    # Export ANativeActivity_onCreate(),
    # Refer to: https://github.com/android-ndk/ndk/issues/381.
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

    add_library(BilliardGame SHARED 
        BilliardGameLogic.cpp BilliardGameConfig.cpp)
ELSEIF(APPLE)
    add_executable(BilliardGame MACOSX_BUNDLE
        BilliardGameLogic.cpp BilliardGameConfig.cpp
        Main.metallib
        )
ELSE(ANDROID)
    add_executable(BilliardGame
        BilliardGameLogic.cpp BilliardGameConfig.cpp)
ENDIF(ANDROID)

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

    add_custom_target(BilliardGameShaderLibrary
        DEPENDS Main.metallib
    )

    add_dependencies(BilliardGame BilliardGameShaderLibrary)

    set(RESOURCE_FILES
        Main.metallib
    )

    set_target_properties(BilliardGame PROPERTIES
        RESOURCE "${RESOURCE_FILES}"
    )

    target_link_libraries(BilliardGame 
                            BulletPhysics
                            Common
                            MetalRHI
                            PlatformInterface
                        )
ELSEIF(WA)
    target_link_libraries(BilliardGame 
                            BulletPhysics
                            OpenGLRHI
                            Common
                            Sdl
                        )
    set_target_properties(BilliardGame
            PROPERTIES LINK_FLAGS "-s FULL_ES2=1 -s FULL_ES3=1 -s USE_SDL=2 -s USE_WEBGL2=1"
            )
ELSE(APPLE)
    target_link_libraries(BilliardGame 
                            BulletPhysics
                            OpenGLRHI
                            Common
                            PlatformInterface
                        )
ENDIF(APPLE)
