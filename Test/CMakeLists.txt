set(TEST_CASES AssetLoaderTest GeomMathTest ColorSpaceConversionTest
               OgexParserTest JpegParserTest PngParserTest DdsParserTest HdrParserTest TgaParserTest
               SceneLoadingTest AnimationTest
               BulletTest NumericalMethodsTest BezierCubic1DTest QuickhullTest GjkTest ChronoTest LinearInterpolateTest QRDecomposeTest PolarDecomposeTest
               RasterizationTest SceneObjectTest
               ASTNodeTest MGEMXParserTest
        )

foreach(TEST_CASE IN LISTS TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    target_link_libraries(${TEST_CASE} Framework)
    add_test(NAME TEST_${TEST_CASE} COMMAND ${TEST_CASE})
endforeach(TEST_CASE)

target_include_directories(MGEMXParserTest PRIVATE ${PROJECT_BINARY_DIR}/Framework/Parser)

IF(WA)
set_target_properties(${TEST_CASES}
        PROPERTIES LINK_FLAGS "--shell-file ${CMAKE_CURRENT_SOURCE_DIR}/Test.html"
        )
ENDIF(WA)

target_link_libraries(BulletTest BulletPhysics)

set(TEST_ASSETS
                Asset/Textures/eye.png 
                Asset/Textures/hdr/PaperMill_posx.dds
                Asset/Textures/hdr/PaperMill_E_3k.hdr
                Asset/Textures/huff_simple0.jpg
                Asset/Textures/interior_lod0.tga
                Asset/Scene/splash.ogex
)

IF(WA)
add_custom_target(PackageAsset ALL
    python $ENV{EMSCRIPTEN}/tools/file_packager.py ${CMAKE_CURRENT_BINARY_DIR}/Asset.data 
        --lz4
        --no-heap-copy
        --js-output=${CMAKE_CURRENT_BINARY_DIR}/Asset.preload.js 
        --preload ${TEST_ASSETS} Asset/Shaders/WebGL
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    VERBATIM
        )
ELSE(WA) # For Windows / MacOS / Linux
    add_custom_target(CopyTestAssets ALL)

    foreach(TEST_ASSET ${TEST_ASSETS})
        add_custom_command(
            TARGET CopyTestAssets
            COMMAND cmake -E copy ${TEST_ASSET} ${CMAKE_CURRENT_BINARY_DIR}/Resources/${TEST_ASSET}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            VERBATIM
        )
    endforeach(TEST_ASSET)
ENDIF(WA)

