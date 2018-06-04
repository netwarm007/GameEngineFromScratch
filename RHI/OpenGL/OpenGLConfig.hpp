#include "OpenGL/OpenGLGraphicsManager.hpp"
#include "OpenGL/OpenGLShaderManager.hpp"

namespace My {
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new OpenGLGraphicsManager);
    IShaderManager*  g_pShaderManager   = static_cast<IShaderManager*>(new OpenGLShaderManager);
}
