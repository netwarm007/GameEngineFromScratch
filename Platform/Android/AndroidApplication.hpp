#pragma once
#include "BaseApplication.hpp"
#include <android/sensor.h>
#include <android/log.h>
#include <android_native_app_glue.h>

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

namespace My {
    class AndroidApplication : public BaseApplication
    {
    public:
        AndroidApplication(GfxConfiguration& cfg);
        virtual int Initialize();
        virtual void Finalize();
        // One cycle of the main loop
        virtual void Tick();

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
}

