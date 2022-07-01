set(FRAMEWORK_TEST_CASES AssetLoaderTest GeomMathTest ColorSpaceConversionTest
               OgexParserTest JpegParserTest PngParserTest DdsParserTest HdrParserTest TgaParserTest
               AstcParserTest PvrParserTest
               SceneLoadingTest AnimationTest
               BulletTest NumericalMethodsTest BezierCubic1DTest QuickhullTest GjkTest ChronoTest LinearInterpolateTest QRDecomposeTest PolarDecomposeTest
               RasterizationTest SceneObjectTest
               ASTNodeTest MGEMXParserTest CodeGeneratorTest
)

foreach(TEST_CASE IN LISTS FRAMEWORK_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    target_link_libraries(${TEST_CASE} Framework PlatformInterface)
    add_test(NAME TEST_${TEST_CASE} COMMAND ${TEST_CASE})
endforeach(TEST_CASE)

target_include_directories(MGEMXParserTest PRIVATE ${PROJECT_BINARY_DIR}/Framework/Parser)
target_include_directories(CodeGeneratorTest PRIVATE ${PROJECT_BINARY_DIR}/Framework/Parser)

if(OS_MACOS OR OS_WINDOWS)
if(Vulkan_FOUND)
    execute_process(COMMAND git submodule update --init ${MYGE_EXTERNAL_ROOT}/../src/tinyobjloader)
    add_executable(VulkanRHITest VulkanRHITest.cpp)
    target_include_directories(VulkanRHITest PRIVATE ${PROJECT_SOURCE_DIR}/External/src/tinyobjloader)
    target_link_libraries(VulkanRHITest Framework VulkanRHI PlatformInterface)
endif(Vulkan_FOUND)
endif(OS_MACOS OR OS_WINDOWS)

if(OS_WINDOWS)
    execute_process(COMMAND git submodule update --init ${MYGE_EXTERNAL_ROOT}/../src/tinyobjloader)
    add_executable(D3d12RHITest D3d12RHITest.cpp)
    target_include_directories(D3d12RHITest PRIVATE ${PROJECT_SOURCE_DIR}/External/src/tinyobjloader)
    target_link_libraries(D3d12RHITest Framework D3dRHI PlatformInterface)
endif(OS_WINDOWS)

set(PLATFORM_TEST_CASES BasicWindowTest)

foreach(TEST_CASE IN LISTS PLATFORM_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    target_link_libraries(${TEST_CASE} Framework PlatformInterface)
endforeach(TEST_CASE)

set(GRAPHICS_TEST_CASES GraphicsWindowTest ImGuiTest)

if(Vulkan_FOUND)
set(VULKAN_RHI VulkanRHI)
endif(Vulkan_FOUND)

foreach(TEST_CASE IN LISTS GRAPHICS_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    if(OS_MACOS)
        target_link_libraries(${TEST_CASE} Framework PlatformInterface MetalRHI ${VULKAN_RHI})
    elseif(OS_WINDOWS)
        target_link_libraries(${TEST_CASE} Framework PlatformInterface D3dRHI OpenGLRHI ${VULKAN_RHI})
    else()
        target_link_libraries(${TEST_CASE} Framework PlatformInterface OpenGLRHI ${VULKAN_RHI})
    endif(OS_MACOS)
endforeach(TEST_CASE)

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

