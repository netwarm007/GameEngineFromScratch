IF(ANDROID OR WA)
add_library(OpenGLRHI
        OpenGLESGraphicsManager.cpp
        OpenGLESPipelineStateManager.cpp
        OpenGLGraphicsManagerCommonBase.cpp
)
ELSE(ANDROID OR WA)
find_package(OpenGL REQUIRED)
add_library(OpenGLRHI
        OpenGLGraphicsManager.cpp
        OpenGLPipelineStateManager.cpp
        OpenGLGraphicsManagerCommonBase.cpp
        ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad.c
        ${PROJECT_SOURCE_DIR}/External/src/imgui/examples/imgui_impl_opengl3.cpp
)
target_link_libraries(OpenGLRHI
        ${OPENGL_LIBRARIES}
)
ENDIF(ANDROID OR WA)
