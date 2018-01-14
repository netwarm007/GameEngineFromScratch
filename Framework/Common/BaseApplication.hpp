#pragma once
#include "IApplication.hpp"
#include "GraphicsManager.hpp"
#include "MemoryManager.hpp"
#include "AssetLoader.hpp"
#include "SceneManager.hpp"
#include "InputManager.hpp"

namespace My {
    class BaseApplication : implements IApplication
    {
    public:
        BaseApplication(GfxConfiguration& cfg);
        virtual int Initialize();
        virtual void Finalize();
        // One cycle of the main loop
        virtual void Tick();

        virtual void SetCommandLineParameters(int argc, char** argv);

        virtual bool IsQuit();

        inline GfxConfiguration& GetConfiguration() { return m_Config; };

        virtual int LoadScene();

        virtual void OnDraw() {};

    protected:
        // Flag if need quit the main loop of the application
        static bool         m_bQuit;
        GfxConfiguration    m_Config;
        int                 m_nArgC;
        char**              m_ppArgV;

    private:
        // hide the default construct to enforce a configuration
        BaseApplication(){};
    };
}

