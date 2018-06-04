#include <iostream>
#include "OpenGLGraphicsManager.hpp"

using namespace My;
using namespace std;

extern struct gladGLversionStruct GLVersion;

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

#include "OpenGLGraphicsManagerCommonBase.cpp"
