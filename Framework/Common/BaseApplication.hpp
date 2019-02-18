#pragma once
#include "IApplication.hpp"

namespace My {
    class BaseApplication : implements IApplication
    {
    public:
        BaseApplication(GfxConfiguration& cfg);
        int Initialize() override;
        void Finalize() override;
        // One cycle of the main loop
        void Tick() override;

        void SetCommandLineParameters(int argc, char** argv) override;
        int  GetCommandLineArgumentsCount() const override;
        const char* GetCommandLineArgument(int index) const override;

        bool IsQuit() const override;
        void RequestQuit() override { m_bQuit = true; }

        void CreateMainWindow() override;

        inline const GfxConfiguration& GetConfiguration() const override { return m_Config; };

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

