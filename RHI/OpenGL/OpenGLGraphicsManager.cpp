#include "OpenGLGraphicsManager.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>

#include "glad/glad.h"

#include "imgui_impl_opengl3.h"
#ifdef OS_WINDOWS
#include "imgui_impl_win32.h"
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

        auto config = m_pApp->GetConfiguration();
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

        if (config.msaaSamples > 1) {
            glEnable(GL_MULTISAMPLE);
        }
    }

    char version[20];
    snprintf(version, 20, "#version %d",
             GLVersion.major * 100 + GLVersion.minor * 10);
    ImGui_ImplOpenGL3_Init(version);

    return result;
}

void OpenGLGraphicsManager::Finalize() {
    ImGui_ImplOpenGL3_Shutdown();
    OpenGLGraphicsManagerCommonBase::Finalize();
}

void OpenGLGraphicsManager::getOpenGLTextureFormat(
    COMPRESSED_FORMAT compressed_format, GLenum& format,
    GLenum& internal_format, GLenum& type) {
    format = GL_COMPRESSED_RGB;

    switch (compressed_format) {
        case COMPRESSED_FORMAT::BC1:
        case COMPRESSED_FORMAT::DXT1:
            internal_format = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::BC2:
        case COMPRESSED_FORMAT::DXT3:
            internal_format = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::BC3:
        case COMPRESSED_FORMAT::DXT5:
            internal_format = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::BC4:
            internal_format = GL_COMPRESSED_RED_RGTC1;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::BC5:
            internal_format = GL_COMPRESSED_RG_RGTC2;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::BC6H:
            internal_format = GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_ARB;
            type = GL_HALF_FLOAT;
            break;
        case COMPRESSED_FORMAT::BC7:
            internal_format = GL_COMPRESSED_RGBA_BPTC_UNORM_ARB;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_4x4:
            internal_format = GL_COMPRESSED_RGBA_ASTC_4x4_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_5x4:
            internal_format = GL_COMPRESSED_RGBA_ASTC_5x4_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_5x5:
            internal_format = GL_COMPRESSED_RGBA_ASTC_5x5_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_6x5:
            internal_format = GL_COMPRESSED_RGBA_ASTC_6x5_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_6x6:
            internal_format = GL_COMPRESSED_RGBA_ASTC_6x6_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_8x5:
            internal_format = GL_COMPRESSED_RGBA_ASTC_8x5_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_8x6:
            internal_format = GL_COMPRESSED_RGBA_ASTC_8x6_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_8x8:
            internal_format = GL_COMPRESSED_RGBA_ASTC_8x8_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        default:
            assert(0);
    }
}

void OpenGLGraphicsManager::getOpenGLTextureFormat(PIXEL_FORMAT pixel_format,
                                               GLenum& format,
                                               GLenum& internal_format,
                                               GLenum& type) {
    switch (pixel_format) {
        case PIXEL_FORMAT::R8:
            format = GL_RED;
            internal_format = GL_R8;
            type = GL_UNSIGNED_BYTE;
            break;
        case PIXEL_FORMAT::RGB8:
            format = GL_RGB;
            internal_format = GL_RGB8;
            type = GL_UNSIGNED_BYTE;
            break;
        case PIXEL_FORMAT::RGBA8:
            format = GL_RGBA;
            internal_format = GL_RGBA8;
            type = GL_UNSIGNED_BYTE;
            break;
        case PIXEL_FORMAT::R16:
            format = GL_RED;
            internal_format = GL_R16F;
            type = GL_HALF_FLOAT;
            break;
        case PIXEL_FORMAT::RG16:
            format = GL_RG;
            internal_format = GL_RG16F;
            type = GL_HALF_FLOAT;
            break;
        case PIXEL_FORMAT::RGB16:
            format = GL_RGB;
            internal_format = GL_RGB16F;
            type = GL_HALF_FLOAT;
            break;
        case PIXEL_FORMAT::RGBA16:
            format = GL_RGBA;
            internal_format = GL_RGBA16F;
            type = GL_HALF_FLOAT;
            break;
        case PIXEL_FORMAT::R32:
            format = GL_RED;
            internal_format = GL_R32F;
            type = GL_FLOAT;
            break;
        case PIXEL_FORMAT::RG32:
            format = GL_RG;
            internal_format = GL_RG32F;
            type = GL_FLOAT;
            break;
        case PIXEL_FORMAT::RGB32:
            format = GL_RGB;
            internal_format = GL_RGB32F;
            type = GL_FLOAT;
            break;
        case PIXEL_FORMAT::RGBA32:
            format = GL_RGBA;
            internal_format = GL_RGBA32F;
            type = GL_FLOAT;
            break;
        case PIXEL_FORMAT::D24R8:
            format = GL_DEPTH;
            internal_format = GL_DEPTH24_STENCIL8;
            type = GL_FLOAT;
            break;
        case PIXEL_FORMAT::D32:
            format = GL_DEPTH;
            internal_format = GL_DEPTH_COMPONENT32;
            type = GL_FLOAT;
            break;
        default:
            assert(0);
    }
}

void OpenGLGraphicsManager::BeginFrame(Frame& frame) {
    OpenGLGraphicsManagerCommonBase::BeginFrame(frame);
    ImGui_ImplOpenGL3_NewFrame();
#ifdef OS_WINDOWS
    ImGui_ImplWin32_NewFrame();
#endif
}

void OpenGLGraphicsManager::EndFrame(Frame& frame) {
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    OpenGLGraphicsManagerCommonBase::EndFrame(frame);
}

void OpenGLGraphicsManager::CreateTextureView(
    Texture2D& texture_view, const TextureArrayBase& texture_array,
    const uint32_t slice, const uint32_t mip) {
    GLuint tex_view = 0;

    if (GLVersion.major >= 4 && GLVersion.minor >= 3) {
        glGenTextures(1, &tex_view);
        glTextureView(tex_view, GL_TEXTURE_2D, (GLuint)texture_array.handler,
                      (GLint)texture_array.format, mip, 1, slice, 1);
    }

    texture_view.handler = tex_view;
    texture_view.format = texture_array.format;
    texture_view.width = texture_array.width;
    texture_view.height = texture_array.height;
}

void OpenGLGraphicsManager::BeginPass(Frame& frame) {
    if (frame.renderToTexture) {
        GLuint frame_buffer;
        // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth
        // buffer.
        glGenFramebuffers(1, &frame_buffer);

        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);

        if (frame.enableMSAA) {
            if (frame.colorTextures[1].handler) {
                //glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                //                    (GLuint)frame.colorTextures[1].handler, 0);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE,
                                      (GLuint)frame.colorTextures[1].handler, 0);
            }
        } else {
            if (frame.colorTextures[0].handler) {
                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                    (GLuint)frame.colorTextures[0].handler, 0);
            }
        }

        if (frame.depthTexture.handler) {
            glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                (GLuint)frame.depthTexture.handler, 0);

        }

        // Always check that our framebuffer is ok
        auto status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            assert(0);
        }

        frame.frameBuffer = frame_buffer;
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // Set viewport
    glViewport(0, 0, m_canvasWidth, m_canvasHeight);

    // Set the color to clear the screen to.
    glClearColor(frame.clearColor[0], frame.clearColor[1], frame.clearColor[2],
                 frame.clearColor[3]);

    // Clear the screen and depth buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLGraphicsManager::EndPass(Frame& frame) {
    if (frame.renderToTexture) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        GLuint frame_buffer = (GLuint)frame.frameBuffer;
        glDeleteFramebuffers(1, &frame_buffer);

        frame.frameBuffer = 0;
    }
}

