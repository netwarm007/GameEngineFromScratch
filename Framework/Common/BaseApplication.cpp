#include "BaseApplication.hpp"
#include <iostream>

using namespace My;

bool My::BaseApplication::m_bQuit = false;

My::BaseApplication::BaseApplication(GfxConfiguration& cfg)
    :m_Config(cfg)
{
}

// Parse command line, read configuration, initialize all sub modules
int My::BaseApplication::Initialize()
{
    int result = 0;

    std::cout << m_Config;

	return result;
}


// Finalize all sub modules and clean up all runtime temporary files.
void My::BaseApplication::Finalize()
{
}


// One cycle of the main loop
void My::BaseApplication::Tick()
{
}

bool My::BaseApplication::IsQuit()
{
	return m_bQuit;
}

