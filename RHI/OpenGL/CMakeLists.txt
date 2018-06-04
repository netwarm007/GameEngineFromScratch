IF(ANDROID)
add_library(OpenGLRHI
        OpenGLESGraphicsManager.cpp
)
ELSE(ANDROID)
add_library(OpenGLRHI
        OpenGLGraphicsManager.cpp
        OpenGLShaderManager.cpp
        ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad.c
)
target_include_directories(OpenGLRHI PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
ENDIF(ANDROID)
target_include_directories(OpenGLRHI PUBLIC GeomMath)
target_link_libraries(OpenGLRHI GeomMath Common)

