#pragma once

#include "BaseApplication.hpp"

namespace My {
    class OrbisApplication : public BaseApplication
    {
        using BaseApplication::BaseApplication;

        virtual int Initialize();
    };
}
