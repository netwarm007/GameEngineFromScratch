add_executable(RayTracingTest RayTracingTest.cpp)
target_link_libraries(RayTracingTest Framework)

# CUDA
if(USE_CUDA)
    set(CUDA_TEST_CASES CudaTest)
    foreach(CUDA_TEST_CASE IN LISTS CUDA_TEST_CASES)
        add_executable(${CUDA_TEST_CASE} ${CUDA_TEST_CASE}.cu)
        #set_property(TARGET RayTracingCUDATest PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(${CUDA_TEST_CASE} Framework)
        add_test(NAME TEST_${CUDA_TEST_CASE} COMMAND ${CUDA_TEST_CASE})
    endforeach()

    set(CUDA_APPS RayTracingCUDATest)
    foreach(CUDA_APP IN LISTS CUDA_APPS)
        add_executable(${CUDA_APP} ${CUDA_APP}.cu)
        target_link_libraries(${CUDA_APP} Framework)
    endforeach()

    find_package(OptiX76)
    if (OptiX76_FOUND)
        add_executable(OptixTest OptixTest.cu ${PROJECT_BINARY_DIR}/Asset/Shaders/CUDA/OptixTest.shader.optixir)
        target_include_directories(OptixTest PUBLIC ${OPTIX76_INCLUDE_DIR})
        target_link_libraries(OptixTest Framework PlatformInterface)

        add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/Asset/Shaders/CUDA/OptixTest.shader.optixir 
            COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/Asset/Shaders/CUDA
            COMMAND nvcc --std c++20 -o ${PROJECT_BINARY_DIR}/Asset/Shaders/CUDA/OptixTest.shader.optixir --optix-ir ${PROJECT_SOURCE_DIR}/Asset/Shaders/CUDA/OptixTest.shader.cu
                    -I ${OPTIX76_INCLUDE_DIR} 
                    -I ${PROJECT_SOURCE_DIR} 
                    -I ${PROJECT_SOURCE_DIR}/Framework/GeomMath/ 
                    -I ${PROJECT_SOURCE_DIR}/Framework/Common/ 
                    -I ${PROJECT_SOURCE_DIR}/Framework/Algorism/ 
                    -I ${PROJECT_SOURCE_DIR}/Test/ 
            DEPENDS ${PROJECT_SOURCE_DIR}/Asset/Shaders/CUDA/OptixTest.shader.cu
        )
    endif()
endif()

IF(WA)
    set_target_properties(${TEST_CASES}
        PROPERTIES LINK_FLAGS "--shell-file ${CMAKE_CURRENT_SOURCE_DIR}/Test.html"
    )
ENDIF(WA)

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

add_subdirectory(Algorism)
add_subdirectory(Audio)
add_subdirectory(Encoder)
add_subdirectory(Framework)
add_subdirectory(Generator)
add_subdirectory(Parser)
add_subdirectory(Platform)
add_subdirectory(RHI)
