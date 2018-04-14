IF(ANDROID)
    # Export ANativeActivity_onCreate(),
    # Refer to: https://github.com/android-ndk/ndk/issues/381.
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

    add_library(BilliardGame SHARED 
        BilliardGameLogic.cpp BilliardGameConfig.cpp)
ELSEIF(APPLE)
    add_executable(BilliardGame MACOSX_BUNDLE
        BilliardGameLogic.cpp BilliardGameConfig.cpp)
ELSE(ANDROID)
    add_executable(BilliardGame 
        BilliardGameLogic.cpp BilliardGameConfig.cpp)
ENDIF(ANDROID)

IF(WIN32)
    target_link_libraries(BilliardGame 
                            BulletPhysics
                            MyGameEngineOpenGL
                        )
    add_custom_command(TARGET BilliardGame POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
            $<TARGET_FILE_DIR:BilliardGame>)
ELSE(WIN32)
    target_link_libraries(BilliardGame 
                            BulletPhysics
                            MyGameEngine
                        )
ENDIF(WIN32)

IF(ANDROID)
    add_custom_command(TARGET BilliardGame POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/Asset/Shaders/OpenGLES/*.glsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/Asset/Shaders")
ELSE(ANDROID)
    add_custom_command(TARGET BilliardGame POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory 
            $<TARGET_FILE_DIR:BilliardGame>/Asset
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/Asset"
            $<TARGET_FILE_DIR:BilliardGame>/Asset)
ENDIF(ANDROID)


