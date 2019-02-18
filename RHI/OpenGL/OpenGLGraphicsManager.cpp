#include <iostream>
#include <sstream>
#include <algorithm>
#include <functional>

#include "OpenGLGraphicsManager.hpp"

#include "glad/glad.h"

extern struct gladGLversionStruct GLVersion;

using namespace My;
using namespace std;

int OpenGLGraphicsManager::Initialize()
{
    int result;

    result = GraphicsManager::Initialize();

    if (result) {
        return result;
    }

    result = gladLoadGL();
    if (!result) {
        cerr << "OpenGL load failed!" << endl;
        result = -1;
    } else {
        result = 0;
        cout << "OpenGL Version " << GLVersion.major << "." << GLVersion.minor << " loaded" << endl;

        if (GLAD_GL_VERSION_3_3) {
            // Set the depth buffer to be entirely cleared to 1.0 values.
            glClearDepth(1.0f);

            // Enable depth testing.
            glEnable(GL_DEPTH_TEST);

            // Set the polygon winding to front facing for the right handed system.
            glFrontFace(GL_CCW);

            // Enable back face culling.
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);

            glEnable(GL_PROGRAM_POINT_SIZE);
        }


        auto config = g_pApp->GetConfiguration();
        glViewport(0, 0, config.screenWidth, config.screenHeight);
    }

    return result;
}

void OpenGLGraphicsManager::getOpenGLTextureFormat(const Image& img, uint32_t& format, uint32_t& internal_format, uint32_t& type)
{
    if(img.compressed)
    {
        format = GL_COMPRESSED_RGB;

        switch (img.compress_format)
        {
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
    }
    else
    {
        if(img.bitcount == 8)
        {
            format = GL_RED;
            internal_format = GL_R8;
            type = GL_UNSIGNED_BYTE;
        }
        else if(img.bitcount == 16)
        {
            format = GL_RED;
            internal_format = GL_R16;
            type = GL_UNSIGNED_SHORT;
        }
        else if(img.bitcount == 24)
        {
            format = GL_RGB;
            internal_format = GL_RGB8;
            type = GL_UNSIGNED_BYTE;
        }
        else if(img.bitcount == 64)
        {
            format = GL_RGBA;
            if (img.is_float)
            {
                internal_format = GL_RGBA16F;
                type = GL_HALF_FLOAT;
            }
            else
            {
                internal_format = GL_RGBA16;
                type = GL_UNSIGNED_SHORT;
            }
        }
        else if(img.bitcount == 128)
        {
            format = GL_RGBA;
            if (img.is_float)
            {
                internal_format = GL_RGBA32F;
                type = GL_FLOAT;
            }
            else
            {
                internal_format = GL_RGBA;
                type = GL_UNSIGNED_INT;
            }
        }
        else
        {
            format = GL_RGBA;
            internal_format = GL_RGBA8;
            type = GL_UNSIGNED_BYTE;
        }
    }
}
