# build native_app_glue as a static lib
add_library(native_app_glue STATIC
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

# Export ANativeActivity_onCreate(),
# Refer to: https://github.com/android-ndk/ndk/issues/381.
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

add_library(PlatformInterface STATIC
        AndroidApplication.cpp 
        AndroidAssetLoader.cpp
        OpenGLESApplication.cpp
        )
target_include_directories(PlatformInterface PRIVATE
    ${ANDROID_NDK}/sources/android/native_app_glue)

target_link_libraries(PlatformInterface
        android
        native_app_glue
        EGL
        GLESv1_CM
        log)

add_library(MyGameEngine 
        AndroidApplicationConfig.cpp
)
target_link_libraries(MyGameEngine
        OpenGLRHI
        PlatformInterface
        Common 
)
target_include_directories(MyGameEngine PRIVATE
    ${ANDROID_NDK}/sources/android/native_app_glue)
