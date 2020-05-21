#pragma once
#include <android/sensor.h>
#include <android_native_app_glue.h>

#include "BaseApplication.hpp"
#include "platform_defs.hpp"

namespace My {
class AndroidApplication : public BaseApplication {
   public:
    using BaseApplication::BaseApplication;
    virtual int Initialize();

    void* GetMainWindowHandler() override { return m_pApp; }

   public:
    /**
     * Our saved state data.
     */
    struct saved_state {
        float angle;
        int32_t x;
        int32_t y;
    };

    /**
     * Shared state for our app.
     */
    struct android_app* m_pApp;

    ASensorManager* m_pSensorManager;
    const ASensor* m_pAccelerometerSensor;
    ASensorEventQueue* m_pSensorEventQueue;

    struct saved_state m_State;
    bool m_bAnimating;
};
}  // namespace My
