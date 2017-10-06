add_executable(TextureLoadTest TextureLoadTest.cpp)
target_link_libraries(TextureLoadTest WindowsApplication Common D2dRHI d2d1)

add_test(NAME TEST_TextureLoad COMMAND TextureLoadTest)

