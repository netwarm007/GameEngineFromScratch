#include "BaseApplication.hpp"

#include <cassert>
#include <iostream>

using namespace My;
using namespace std;

bool BaseApplication::m_bQuit = false;

BaseApplication::BaseApplication(GfxConfiguration& cfg) : m_Config(cfg) {}

// Parse command line, read configuration, initialize all sub modules
int BaseApplication::Initialize() {
    int ret = 0;

    cout << m_Config;

    // create the main window
    CreateMainWindow();

    return ret;
}

// Finalize all sub modules and clean up all runtime temporary files.
void BaseApplication::Finalize() {}

// One cycle of the main loop
void BaseApplication::Tick() {}

void BaseApplication::SetCommandLineParameters(int argc, char** argv) {
    m_nArgC = argc;
    m_ppArgV = argv;
}

int BaseApplication::GetCommandLineArgumentsCount() const { return m_nArgC; }

const char* BaseApplication::GetCommandLineArgument(int index) const {
    assert(index < m_nArgC);
    return m_ppArgV[index];
}

void BaseApplication::CreateMainWindow() {}

bool BaseApplication::IsQuit() const { return m_bQuit; }
