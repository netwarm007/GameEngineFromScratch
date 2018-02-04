#include <cstdio>
#include "OpenGLESGraphicsManager.hpp"

using namespace My;
using namespace std;

int OpenGLESGraphicsManager::Initialize()
{
    int result = 0;

    auto opengl_info = {GL_VENDOR, GL_RENDERER, GL_VERSION, GL_EXTENSIONS};
    for (auto name : opengl_info) {
        auto info = glGetString(name);
        printf("OpenGL Info: %s", info);
    }

    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
    glEnable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);
    glDisable(GL_DEPTH_TEST);

    return result;
}

void OpenGLESGraphicsManager::Finalize()
{
}

void OpenGLESGraphicsManager::Clear()
{
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void OpenGLESGraphicsManager::Draw()
{
    glFlush();
}

