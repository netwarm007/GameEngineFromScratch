add_executable(TextureLoadTest TextureLoadTest.cpp)
target_link_libraries(TextureLoadTest WindowsApplication Common ${OPENGEX_LIBRARY} ${OPENDDL_LIBRARY} ${XG_LIBRARY} D2dRHI d2d1)
__add_xg_platform_dependencies(TextureLoadTest)

add_test(NAME TEST_TextureLoad COMMAND TextureLoadTest)

