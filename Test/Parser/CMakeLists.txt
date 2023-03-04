set(PARSER_TEST_CASES 
    AstcParserTest
    DdsParserTest
    HdrParserTest
    JpegParserTest
    MGEMXParserTest
    OgexParserTest
    PngParserTest
    PvrParserTest
    TgaParserTest
    WaveParserTest
)

foreach(TEST_CASE IN LISTS PARSER_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    target_link_libraries(${TEST_CASE} Framework PlatformInterface)
    add_test(NAME TEST_${TEST_CASE} COMMAND ${TEST_CASE})
endforeach()

target_include_directories(MGEMXParserTest PRIVATE ${PROJECT_BINARY_DIR}/Framework/Parser)
