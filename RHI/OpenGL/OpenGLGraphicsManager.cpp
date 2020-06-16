#include "OpenGLGraphicsManager.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>

#include "glad/glad.h"

#include "imgui/examples/imgui_impl_opengl3.h"
#ifdef OS_WINDOWS
#include "imgui/examples/imgui_impl_win32.h"
#endif

using namespace My;
using namespace std;

static void APIENTRY glDebugOutput(GLenum source, GLenum type, unsigned int id,
                                   GLenum severity, GLsizei length,
                                   const char* message, const void* userParam) {
    // ignore non-significant error/warning codes
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

    std::cerr << "---------------" << std::endl;
    std::cerr << "Debug message (" << id << "): " << message << std::endl;

    switch (source) {
        case GL_DEBUG_SOURCE_API:
            std::cout << "Source: API";
            break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
            std::cout << "Source: Window System";
            break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER:
            std::cout << "Source: Shader Compiler";
            break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:
            std::cout << "Source: Third Party";
            break;
        case GL_DEBUG_SOURCE_APPLICATION:
            std::cout << "Source: Application";
            break;
        case GL_DEBUG_SOURCE_OTHER:
            std::cout << "Source: Other";
            break;
    }
    std::cerr << std::endl;

    switch (type) {
        case GL_DEBUG_TYPE_ERROR:
            std::cout << "Type: Error";
            break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
            std::cout << "Type: Deprecated Behaviour";
            break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
            std::cout << "Type: Undefined Behaviour";
            break;
        case GL_DEBUG_TYPE_PORTABILITY:
            std::cout << "Type: Portability";
            break;
        case GL_DEBUG_TYPE_PERFORMANCE:
            std::cout << "Type: Performance";
            break;
        case GL_DEBUG_TYPE_MARKER:
            std::cout << "Type: Marker";
            break;
        case GL_DEBUG_TYPE_PUSH_GROUP:
            std::cout << "Type: Push Group";
            break;
        case GL_DEBUG_TYPE_POP_GROUP:
            std::cout << "Type: Pop Group";
            break;
        case GL_DEBUG_TYPE_OTHER:
            std::cout << "Type: Other";
            break;
    }
    std::cerr << std::endl;

    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:
            std::cout << "Severity: high";
            break;
        case GL_DEBUG_SEVERITY_MEDIUM:
            std::cout << "Severity: medium";
            break;
        case GL_DEBUG_SEVERITY_LOW:
            std::cout << "Severity: low";
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            std::cout << "Severity: notification";
            break;
    }
    std::cerr << std::endl;
    std::cerr << std::endl;
}

int OpenGLGraphicsManager::Initialize() {
    int result;

    result = OpenGLGraphicsManagerCommonBase::Initialize();

    if (result) {
        return result;
    }

    result = gladLoadGL();
    if (!result) {
        cerr << "OpenGL load failed!" << endl;
        result = -1;
    } else {
        result = 0;
        cout << "OpenGL Version " << GLVersion.major << "." << GLVersion.minor
             << " loaded" << endl;

        if (GLAD_GL_VERSION_3_3) {
            // Set the depth buffer to be entirely cleared to 1.0 values.
            glClearDepth(1.0f);

            // Enable depth testing.
            glEnable(GL_DEPTH_TEST);

            // Set the polygon winding to front facing for the right handed
            // system.
            glFrontFace(GL_CCW);

            // Enable back face culling.
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);

            glEnable(GL_PROGRAM_POINT_SIZE);
        }

        auto config = g_pApp->GetConfiguration();
        glViewport(0, 0, config.screenWidth, config.screenHeight);

        int flags;
        glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
        if (flags & GL_CONTEXT_FLAG_DEBUG_BIT) {
            cout << "OpenGL Debug Context Enabled" << endl;
            glEnable(GL_DEBUG_OUTPUT);
            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            glDebugMessageCallback(glDebugOutput, nullptr);
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0,
                                  nullptr, GL_TRUE);
        }
    }

    ImGui_ImplOpenGL3_Init("#version 420");

    return result;
}

void OpenGLGraphicsManager::Finalize() {
    ImGui_ImplOpenGL3_Shutdown();
    OpenGLGraphicsManagerCommonBase::Finalize();
}

void OpenGLGraphicsManager::getOpenGLTextureFormat(const Image& img,
                                                   uint32_t& format,
                                                   uint32_t& internal_format,
                                                   uint32_t& type) {
    if (img.compressed) {
        format = GL_COMPRESSED_RGB;

        switch (img.compress_format) {
            case "DXT1"_u32:
                internal_format = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
                break;
            case "DXT3"_u32:
                internal_format = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
                break;
            case "DXT5"_u32:
                internal_format = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
                break;
            default:
                assert(0);
        }

        type = GL_UNSIGNED_BYTE;
    } else {
        if (img.bitcount == 8) {
            format = GL_RED;
            internal_format = GL_R8;
            type = GL_UNSIGNED_BYTE;
        } else if (img.bitcount == 16) {
            format = GL_RED;
            internal_format = GL_R16;
            type = GL_UNSIGNED_SHORT;
        } else if (img.bitcount == 24) {
            format = GL_RGB;
            internal_format = GL_RGB8;
            type = GL_UNSIGNED_BYTE;
        } else if (img.bitcount == 64) {
            format = GL_RGBA;
            if (img.is_float) {
                internal_format = GL_RGBA16F;
                type = GL_HALF_FLOAT;
            } else {
                internal_format = GL_RGBA16;
                type = GL_UNSIGNED_SHORT;
            }
        } else if (img.bitcount == 128) {
            format = GL_RGBA;
            if (img.is_float) {
                internal_format = GL_RGBA32F;
                type = GL_FLOAT;
            } else {
                internal_format = GL_RGBA;
                type = GL_UNSIGNED_INT;
            }
        } else {
            format = GL_RGBA;
            internal_format = GL_RGBA8;
            type = GL_UNSIGNED_BYTE;
        }
    }
}

void OpenGLGraphicsManager::BeginFrame(const Frame& frame) {
    OpenGLGraphicsManagerCommonBase::BeginFrame(frame);
    ImGui_ImplOpenGL3_NewFrame();
#ifdef OS_WINDOWS
    ImGui_ImplWin32_NewFrame();
#endif
}

void OpenGLGraphicsManager::EndFrame(const Frame& frame) {
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    OpenGLGraphicsManagerCommonBase::EndFrame(frame);
}