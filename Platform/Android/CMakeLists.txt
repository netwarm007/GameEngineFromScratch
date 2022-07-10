add_library(PlatformInterface STATIC
        $ENV{ANDROID_NDK_ROOT}/sources/android/native_app_glue/android_native_app_glue.c
        AndroidApplication.cpp 
        AndroidAssetLoader.cpp
        OpenGLESApplication.cpp
        )
target_include_directories(PlatformInterface PUBLIC
    $ENV{ANDROID_NDK_ROOT}/sources/android/native_app_glue)

target_link_libraries(PlatformInterface
        android
        EGL
        GLESv3
        log)
