#include "OpenGLESGraphicsManager.hpp"

#include <GLES3/gl32.h>

#include <GLES2/gl2ext.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>

#if defined(OS_WEBASSEMBLY)
// disable compute shader
#define GLAD_GL_ARB_compute_shader 0
#endif

using namespace My;
using namespace std;

int OpenGLESGraphicsManager::Initialize() {
    int result;

    result = GraphicsManager::Initialize();

    if (result) {
        return result;
    }

#if 0
    auto opengl_info = {GL_VENDOR, GL_RENDERER, GL_VERSION, GL_EXTENSIONS};
    for (auto name : opengl_info) {
        auto info = glGetString(name);
        printf("OpenGL Info: %s", info);
    }
#endif

    // Set the depth buffer to be entirely cleared to 1.0 values.
    glClearDepthf(1.0f);

    // Enable depth testing.
    glEnable(GL_DEPTH_TEST);

    // Set the polygon winding to front facing for the right handed system.
    glFrontFace(GL_CCW);

    // Enable back face culling.
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    auto config = m_pApp->GetConfiguration();
    glViewport(0, 0, config.screenWidth, config.screenHeight);

    return result;
}

void OpenGLESGraphicsManager::getOpenGLTextureFormat(
    COMPRESSED_FORMAT compressed_format, GLenum& format,
    GLenum& internal_format, GLenum& type) {

    switch (compressed_format) {
        case COMPRESSED_FORMAT::ETC:
            format = GL_RGB;
            internal_format = GL_COMPRESSED_RGBA8_ETC2_EAC;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_4x4:
            format = GL_RGBA;
            internal_format = GL_COMPRESSED_RGBA_ASTC_4x4_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_5x4:
            format = GL_RGBA;
            internal_format = GL_COMPRESSED_RGBA_ASTC_5x4_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_5x5:
            format = GL_RGBA;
            internal_format = GL_COMPRESSED_RGBA_ASTC_5x5_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_6x5:
            format = GL_RGBA;
            internal_format = GL_COMPRESSED_RGBA_ASTC_6x5_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_6x6:
            format = GL_RGBA;
            internal_format = GL_COMPRESSED_RGBA_ASTC_6x6_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_8x5:
            format = GL_RGBA;
            internal_format = GL_COMPRESSED_RGBA_ASTC_8x5_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_8x6:
            format = GL_RGBA;
            internal_format = GL_COMPRESSED_RGBA_ASTC_8x6_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        case COMPRESSED_FORMAT::ASTC_8x8:
            format = GL_RGBA;
            internal_format = GL_COMPRESSED_RGBA_ASTC_8x8_KHR;
            type = GL_UNSIGNED_BYTE;
            break;
        default:
            assert(0);
    }
}

void OpenGLESGraphicsManager::getOpenGLTextureFormat(PIXEL_FORMAT pixel_format,
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
            internal_format = GL_DEPTH_COMPONENT32F;
            type = GL_FLOAT;
            break;
        default:
            assert(0);
    }
}
