#include <iostream>
#include "OpenGLESGraphicsManager.hpp"

using namespace My;
using namespace std;

int OpenGLESGraphicsManager::Initialize()
{
    int result;

    result = GraphicsManager::Initialize();

    if (result) {
        return result;
    }

    auto opengl_info = {GL_VENDOR, GL_RENDERER, GL_VERSION, GL_EXTENSIONS};
    for (auto name : opengl_info) {
        auto info = glGetString(name);
        printf("OpenGL Info: %s", info);
    }

	// Set the depth buffer to be entirely cleared to 1.0 values.
	glClearDepthf(1.0f);

	// Enable depth testing.
	glEnable(GL_DEPTH_TEST);

	// Set the polygon winding to front facing for the right handed system.
	glFrontFace(GL_CCW);

	// Enable back face culling.
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

    auto config = g_pApp->GetConfiguration();
    glViewport(0, 0, config.screenWidth, config.screenHeight);

    return result;
}

#define OPENGL_ES
#include "OpenGLGraphicsManagerCommonBase.cpp"
#undef OPENGL_ES
