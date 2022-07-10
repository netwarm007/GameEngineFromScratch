# build native_app_glue as a static lib
add_library(native_app_glue STATIC
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

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
        GLESv3
        log)
