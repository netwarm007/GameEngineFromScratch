set(GENERATOR_TEST_CASES 
    CodeGeneratorTest
    WaveGeneratorTest
)

foreach(TEST_CASE IN LISTS GENERATOR_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    target_link_libraries(${TEST_CASE} Framework PlatformInterface)
    add_test(NAME TEST_${TEST_CASE} COMMAND ${TEST_CASE})
endforeach()

target_include_directories(CodeGeneratorTest PRIVATE ${PROJECT_BINARY_DIR}/Framework/Parser)
