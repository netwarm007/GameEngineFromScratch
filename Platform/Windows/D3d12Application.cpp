#include <tchar.h>
#include "D3d12Application.hpp"
#include "D3d/D3d12GraphicsManager.hpp"

using namespace My;

void D3d12Application::Tick()
{
    WindowsApplication::Tick();
    
    // Present the back buffer to the screen since rendering is complete.
    SwapBuffers(m_hDc);
}

