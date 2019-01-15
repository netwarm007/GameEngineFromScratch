IF(ANDROID)
    # Export ANativeActivity_onCreate(),
    # Refer to: https://github.com/android-ndk/ndk/issues/381.
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

    add_library(BilliardGame SHARED 
        BilliardGameLogic.cpp BilliardGameConfig.cpp)
ELSE(ANDROID)
    add_executable(BilliardGame MACOSX_BUNDLE
        BilliardGameLogic.cpp BilliardGameConfig.cpp)
ENDIF(ANDROID)

IF(APPLE)
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

IF(WIN32)
    add_custom_command(TARGET BilliardGame POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
            $<TARGET_FILE_DIR:BilliardGame>)
ENDIF(WIN32)

target_include_directories(BilliardGame PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
