set(FRAMEWORK_TEST_CASES 
    AnimationTest
    AssetLoaderTest 
    GeomMathTest
    GuiSubPassLayoutTest
    SceneGraphTest
    SceneLoadingTest 
    SceneObjectTest
    SceneObjectTextureTest
)

foreach(TEST_CASE IN LISTS FRAMEWORK_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    target_link_libraries(${TEST_CASE} Framework PlatformInterface)
    add_test(NAME TEST_${TEST_CASE} COMMAND ${TEST_CASE})
endforeach()

add_executable(SceneServicesTest SceneServicesTest.cpp)
target_link_libraries(SceneServicesTest PRIVATE Framework PlatformInterface)
add_test(NAME TEST_SceneServicesTest COMMAND SceneServicesTest)
