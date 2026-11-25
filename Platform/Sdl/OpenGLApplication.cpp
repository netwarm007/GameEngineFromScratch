#include "OpenGLApplication.hpp"

using namespace My;

OpenGLApplication::OpenGLApplication()
    : SdlApplication(),
      m_Backend(*this, *this, [this] { return GetSdlWindow(); },
                [this](Uint32 flags) { SetWindowFlags(flags); }) {}

OpenGLApplication::OpenGLApplication(GfxConfiguration& cfg)
    : SdlApplication(cfg),
      m_Backend(*this, *this, [this] { return GetSdlWindow(); },
                [this](Uint32 flags) { SetWindowFlags(flags); }) {}
