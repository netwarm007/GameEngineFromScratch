IF(ANDROID)
add_library(OpenGLRHI
        OpenGLESGraphicsManager.cpp
)
ELSE()
add_library(OpenGLRHI
        OpenGLGraphicsManager.cpp
        ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad.c
)
target_include_directories(OpenGLRHI PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
ENDIF(ANDROID)
add_dependencies(OpenGLRHI GeomMath)

