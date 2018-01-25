add_executable(TextureLoadTest TextureLoadTest.cpp)
target_link_libraries(TextureLoadTest 
                        Common 
                        D2dRHI
                        PlatformInterface
)
__add_xg_platform_dependencies(TextureLoadTest)

add_test(NAME TEST_TextureLoad COMMAND TextureLoadTest)

