add_executable(AssetLoaderTest AssetLoaderTest.cpp)
target_link_libraries(AssetLoaderTest Common)
add_test(NAME TEST_AssetLoader COMMAND AssetLoaderTest)

add_executable(GeomMathTest GeomMathTest.cpp)
target_link_libraries(GeomMathTest GeomMath)
add_test(NAME Test_GeomMath COMMAND GeomMathTest)

add_executable(ColorSpaceConversionTest ColorSpaceConversionTest.cpp)
target_link_libraries(ColorSpaceConversionTest GeomMath)
add_test(NAME Test_ColorSpaceConversion COMMAND ColorSpaceConversionTest)

add_executable(OgexParserTest OgexParserTest.cpp)
target_link_libraries(OgexParserTest Common)
add_test(NAME TEST_OgexParser COMMAND OgexParserTest)

add_executable(JpegParserTest JpegParserTest.cpp)
target_link_libraries(JpegParserTest Common)
add_test(NAME TEST_JpegParser COMMAND JpegParserTest)

add_executable(PngParserTest PngParserTest.cpp)
target_link_libraries(PngParserTest Common)
add_test(NAME TEST_PngParser COMMAND PngParserTest)

add_executable(DdsParserTest DdsParserTest.cpp)
target_link_libraries(DdsParserTest Common)
add_test(NAME TEST_DdsParser COMMAND DdsParserTest)

add_executable(HdrParserTest HdrParserTest.cpp)
target_link_libraries(HdrParserTest Common)
add_test(NAME TEST_HdrParser COMMAND HdrParserTest)

IF(WIN32)
add_custom_command(TARGET PngParserTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
        $<TARGET_FILE_DIR:PngParserTest>)
ENDIF(WIN32)

add_executable(TgaParserTest TgaParserTest.cpp)
target_link_libraries(TgaParserTest Common)
add_test(NAME TEST_TgaParser COMMAND TgaParserTest)

#add_executable(SceneObjectTest SceneObjectTest.cpp)
#target_link_libraries(SceneObjectTest Common ${XG_LIBRARY})
#__add_xg_platform_dependencies(SceneObjectTest)
#add_test(NAME TEST_SceneObject COMMAND SceneObjectTest)

add_executable(SceneLoadingTest SceneLoadingTest.cpp)
target_link_libraries(SceneLoadingTest Common)
add_test(NAME TEST_SceneLoading COMMAND SceneLoadingTest)

add_executable(AnimationTest AnimationTest.cpp)
target_link_libraries(AnimationTest Common)
#add_test(NAME TEST_Animation COMMAND AnimationTest)

add_executable(BulletTest BulletTest.cpp)
target_link_libraries(BulletTest BulletPhysics)
add_test(NAME TEST_Bullet COMMAND BulletTest)

add_executable(NumericalMethodsTest NumericalMethodsTest.cpp)
add_test(NAME TEST_NumericalMethods COMMAND NumericalMethodsTest)

add_executable(BezierCubic1DTest BezierCubic1DTest.cpp)
add_test(NAME TEST_BezierCubic1D COMMAND BezierCubic1DTest)
target_include_directories(BezierCubic1DTest PUBLIC GeomMath)
target_link_libraries(BezierCubic1DTest GeomMath)

add_executable(QuickhullTest QuickhullTest.cpp)
target_link_libraries(QuickhullTest Algorism Geometries GeomMath)
add_test(NAME TEST_Quickhull COMMAND QuickhullTest)

add_executable(GjkTest GjkTest.cpp)
target_link_libraries(GjkTest Algorism Geometries GeomMath)
add_test(NAME TEST_Gjk COMMAND GjkTest)

add_executable(ChronoTest ChronoTest.cpp)
add_test(NAME TEST_Chrono COMMAND ChronoTest)

add_executable(LinearInterpolateTest LinearInterpolateTest.cpp)
target_link_libraries(LinearInterpolateTest Algorism GeomMath)
add_test(NAME TEST_LinearInterpolate COMMAND LinearInterpolateTest)

add_executable(QRDecomposeTest QRDecomposeTest.cpp)
target_link_libraries(QRDecomposeTest Algorism GeomMath)
add_test(NAME TEST_QRDecompose COMMAND QRDecomposeTest)

add_executable(PolarDecomposeTest PolarDecomposeTest.cpp)
target_link_libraries(PolarDecomposeTest Algorism GeomMath)
add_test(NAME TEST_PolarDecompose COMMAND PolarDecomposeTest)

add_executable(RasterizationTest RasterizationTest.cpp)
target_link_libraries(RasterizationTest Algorism GeomMath)
#add_test(NAME TEST_Rasterization COMMAND RasterizationTest)
