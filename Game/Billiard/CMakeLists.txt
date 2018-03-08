IF(ANDROID)
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


