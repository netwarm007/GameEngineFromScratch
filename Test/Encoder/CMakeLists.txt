set(ENCODER_TEST_CASES 
    PPMEncoderTest
    TgaEncoderTest
)

foreach(TEST_CASE IN LISTS ENCODER_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    add_test(NAME TEST_${TEST_CASE} COMMAND ${TEST_CASE})
endforeach()

