add_platform_asset_io(PlatformAssetIO AndroidAssetLoader.cpp)
target_link_libraries(PlatformAssetIO INTERFACE android log)

add_library(PlatformInterface STATIC
        $ENV{ANDROID_NDK_ROOT}/sources/android/native_app_glue/android_native_app_glue.c
        AndroidApplication.cpp 
        OpenGLESApplication.cpp
        )
target_include_directories(PlatformInterface PUBLIC
    $ENV{ANDROID_NDK_ROOT}/sources/android/native_app_glue)

target_link_libraries(PlatformInterface
        PlatformAssetIO
        android
        EGL
        GLESv3
        log)
