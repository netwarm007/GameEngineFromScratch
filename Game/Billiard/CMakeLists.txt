add_executable(BilliardGame BilliardGameLogic.cpp BilliardGameConfig.cpp)
target_link_libraries(BilliardGame 
                        Common 
                        PlatformInterface
                        OpenGLRHI
                     )
target_include_directories(BilliardGame PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
__add_xg_platform_dependencies(BilliardGame)