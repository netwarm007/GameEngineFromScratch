set(ALGORISM_TEST_CASES 
    ASTNodeTest
    BezierCubic1DTest
    BulletTest
    ChronoTest
    ColorSpaceConversionTest
    GjkTest
    LinearInterpolateTest
    NumericalMethodsTest
    PolarDecomposeTest
    QRDecomposeTest
    QuickhullTest
    RandomTest
    RasterizationTest
)

foreach(TEST_CASE IN LISTS ALGORISM_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    add_test(NAME TEST_${TEST_CASE} COMMAND ${TEST_CASE})
endforeach()

target_link_libraries(BulletTest BulletPhysics)
