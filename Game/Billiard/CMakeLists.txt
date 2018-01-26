IF(ANDROID)
add_library(BilliardGame SHARED BilliardGameLogic.cpp BilliardGameConfig.cpp)
ELSE(ANDROID)
add_executable(BilliardGame BilliardGameLogic.cpp BilliardGameConfig.cpp)
ENDIF(ANDROID)
target_link_libraries(BilliardGame 
                        OpenGLRHI
                        PlatformInterface
                        Common 
                     )
target_include_directories(BilliardGame PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
__add_xg_platform_dependencies(BilliardGame)
IF(ANDROID)
target_include_directories(BilliardGame PRIVATE
    ${ANDROID_NDK}/sources/android/native_app_glue)
ENDIF(ANDROID)