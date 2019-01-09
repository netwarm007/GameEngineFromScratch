#include "OpenGLApplication.hpp"
#include <cstdio>

using namespace std;
using namespace My;

void OpenGLApplication::CreateMainWindow()
{
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, m_Config.redBits);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, m_Config.blueBits);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, m_Config.greenBits);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, m_Config.alphaBits);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, m_Config.depthBits);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, m_Config.stencilBits);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, m_Config.msaaSamples);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    m_pWindow = SDL_CreateWindow(m_Config.appName, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, m_Config.screenWidth, m_Config.screenHeight, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (m_pWindow == nullptr){
        logSDLError(std::cout, "CreateWindow");
        SDL_Quit();
    }

	SDL_GL_CreateContext(m_pWindow);

    int major_ver, minor_ver;
    int r, g, b, a, d, s;
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &major_ver);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &minor_ver);
    SDL_GL_GetAttribute(SDL_GL_RED_SIZE, &r);
    SDL_GL_GetAttribute(SDL_GL_GREEN_SIZE, &g);
    SDL_GL_GetAttribute(SDL_GL_BLUE_SIZE, &b);
    SDL_GL_GetAttribute(SDL_GL_ALPHA_SIZE, &a);
    SDL_GL_GetAttribute(SDL_GL_DEPTH_SIZE, &d);
    SDL_GL_GetAttribute(SDL_GL_STENCIL_SIZE, &s);

    printf("OpenGL Version: %d.%d\n", major_ver, minor_ver);
    printf("Red size: %d, Green size: %d, Blue size: %d, Alpha size: %d\n", r, g, b, a);
    printf("Depth size: %d, Stencil size: %d\n", d, s);

	SDL_GL_SetSwapInterval(1);
}

void OpenGLApplication::Tick()
{
    SdlApplication::Tick();
    SDL_GL_SwapWindow(m_pWindow);
}
