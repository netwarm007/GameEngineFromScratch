set(FRAMEWORK_TEST_CASES 
    AnimationTest
    AssetLoaderTest 
    GeomMathTest
    SceneLoadingTest 
    SceneObjectTest
)

foreach(TEST_CASE IN LISTS FRAMEWORK_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    target_link_libraries(${TEST_CASE} Framework PlatformInterface)
    add_test(NAME TEST_${TEST_CASE} COMMAND ${TEST_CASE})
endforeach()

