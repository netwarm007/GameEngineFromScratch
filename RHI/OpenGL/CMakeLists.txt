IF(ANDROID OR WA)
add_library(OpenGLRHI
        OpenGLESGraphicsManager.cpp
        OpenGLESShaderManager.cpp
        OpenGLGraphicsManagerCommonBase.cpp
)
ELSE(ANDROID OR WA)
find_package(OpenGL REQUIRED)
add_library(OpenGLRHI
        OpenGLGraphicsManager.cpp
        OpenGLShaderManager.cpp
        OpenGLGraphicsManagerCommonBase.cpp
        ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad.c
)
target_include_directories(OpenGLRHI PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
target_link_libraries(OpenGLRHI
        ${OPENGL_LIBRARIES}
)
ENDIF(ANDROID OR WA)
