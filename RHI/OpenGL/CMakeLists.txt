add_library(OpenGLRHI
        OpenGLGraphicsManager.cpp
        OpenGLESGraphicsManager.cpp
        ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad.c
)
target_include_directories(OpenGLRHI PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
add_dependencies(OpenGLRHI GeomMath)

