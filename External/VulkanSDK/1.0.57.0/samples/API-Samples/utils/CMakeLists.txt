file(GLOB UTILS_SOURCE *.cpp)

set(SAMPLES_DATA_DIR ${SAMPLES_DATA_DIR} "${PROJECT_SOURCE_DIR}/API-Samples/data")
if(SDK_INCLUDE_PATH)
    include_directories( ${SAMPLES_DATA_DIR} ${GLSLANG_SPIRV_INCLUDE_DIR} ${GLMINC_PREFIX} ${SDK_INCLUDE_PATH} )
else()
    include_directories( ${SAMPLES_DATA_DIR} ${GLSLANG_SPIRV_INCLUDE_DIR} ${GLMINC_PREFIX} )
endif()

if(ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -Werror -DVK_USE_PLATFORM_ANDROID_KHR")
elseif(WIN32)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES")

    # If MSVC, disable some signed/unsigned mismatch warnings.
    if (MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4267")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267")
    endif()
else()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
endif()

add_library(${UTILS_NAME} STATIC ${UTILS_SOURCE})

if(ANDROID)
   add_library(native_app_glue STATIC
               ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
   target_include_directories(${UTILS_NAME} PRIVATE
                              ${ANDROID_NDK}/sources/android/native_app_glue
                              ${CMAKE_CURRENT_SOURCE_DIR}/../android/vulkan_wrapper
                              ${CMAKE_CURRENT_SOURCE_DIR}/../data
                              ${ANDROID_NDK}/sources/third_party/shaderc/include)
   target_link_libraries(${UTILS_NAME}
                              android
                              log
                              native_app_glue)
endif()

