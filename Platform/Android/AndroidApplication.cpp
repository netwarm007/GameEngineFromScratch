#include <unistd.h>
#include <thread>
#include <dlfcn.h>
#include "AndroidApplication.hpp"
#include "AndroidAssetLoader.hpp"

using namespace My;
using namespace std;

AndroidApplication::AndroidApplication(GfxConfiguration& cfg) 
        : BaseApplication(cfg)
{
}

int AndroidApplication::Initialize()
{
    dynamic_cast<AndroidAssetLoader*>(g_pAssetLoader)->SetPlatformAssetManager(m_pApp->activity->assetManager);
    int result = BaseApplication::Initialize();
    if (result)
        LOGE("AndroidApplication Initialize Failed!");

    return result;
}

void AndroidApplication::Finalize()
{
    BaseApplication::Finalize();
}

void AndroidApplication::Tick()
{
    BaseApplication::Tick();
}

/**
 * Process the next input event.
 */
static int32_t engine_handle_input(struct android_app* app, AInputEvent* event) {
    AndroidApplication* engine = (AndroidApplication*)app->userData;
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
        engine->m_bAnimating = true;
        engine->m_State.x = AMotionEvent_getX(event, 0);
        engine->m_State.y = AMotionEvent_getY(event, 0);
        return 1;
    }
    return 0;
}

/**
 * Process the next main command.
 */
static void engine_handle_cmd(struct android_app* app, int32_t cmd) {
    AndroidApplication* engine = (AndroidApplication*)app->userData;
    switch (cmd) {
        case APP_CMD_SAVE_STATE:
            // The system has asked us to save our current state.  Do so.
            engine->m_pApp->savedState = malloc(sizeof(AndroidApplication::saved_state));
            *((AndroidApplication::saved_state*)engine->m_pApp->savedState) = engine->m_State;
            engine->m_pApp->savedStateSize = sizeof(AndroidApplication::saved_state);
            break;
        case APP_CMD_INIT_WINDOW:
            // The window is being shown, get it ready.
            if (engine->m_pApp->window != NULL) {
                auto err = engine->Initialize();
                if (err) {
                    LOGF("Engine Initialize Failed. error code = %d", err);
                }
            }
            break;
        case APP_CMD_TERM_WINDOW:
            // The window is being hidden or closed, clean it up.
            break;
        case APP_CMD_GAINED_FOCUS:
            // When our app gains focus, we start monitoring the accelerometer.
            if (engine->m_pAccelerometerSensor != NULL) {
                ASensorEventQueue_enableSensor(engine->m_pSensorEventQueue,
                                               engine->m_pAccelerometerSensor);
                // We'd like to get 60 events per second (in us).
                ASensorEventQueue_setEventRate(engine->m_pSensorEventQueue,
                                               engine->m_pAccelerometerSensor,
                                               (1000L/60)*1000);
            }
            break;
        case APP_CMD_LOST_FOCUS:
            // When our app loses focus, we stop monitoring the accelerometer.
            // This is to avoid consuming battery while not being used.
            if (engine->m_pAccelerometerSensor != NULL) {
                ASensorEventQueue_disableSensor(engine->m_pSensorEventQueue,
                                                engine->m_pAccelerometerSensor);
            }
            // Also stop animating.
            engine->m_bAnimating = false;
            break;
    }
}

/*
 * AcquireASensorManagerInstance(void)
 *    Workaround ASensorManager_getInstance() deprecation false alarm
 *    for Android-N and before, when compiling with NDK-r15
 */
#include <dlfcn.h>
ASensorManager* AcquireASensorManagerInstance(android_app* app) {

  if(!app)
    return nullptr;

  typedef ASensorManager *(*PF_GETINSTANCEFORPACKAGE)(const char *name);
  void* androidHandle = dlopen("libandroid.so", RTLD_NOW);
  PF_GETINSTANCEFORPACKAGE getInstanceForPackageFunc = (PF_GETINSTANCEFORPACKAGE)
      dlsym(androidHandle, "ASensorManager_getInstanceForPackage");
  if (getInstanceForPackageFunc) {
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, NULL);

    jclass android_content_Context = env->GetObjectClass(app->activity->clazz);
    jmethodID midGetPackageName = env->GetMethodID(android_content_Context,
                                                   "getPackageName",
                                                   "()Ljava/lang/String;");
    jstring packageName= (jstring)env->CallObjectMethod(app->activity->clazz,
                                                        midGetPackageName);

    const char *nativePackageName = env->GetStringUTFChars(packageName, 0);
    ASensorManager* mgr = getInstanceForPackageFunc(nativePackageName);
    env->ReleaseStringUTFChars(packageName, nativePackageName);
    app->activity->vm->DetachCurrentThread();
    if (mgr) {
      dlclose(androidHandle);
      return mgr;
    }
  }

  typedef ASensorManager *(*PF_GETINSTANCE)();
  PF_GETINSTANCE getInstanceFunc = (PF_GETINSTANCE)
      dlsym(androidHandle, "ASensorManager_getInstance");
  // by all means at this point, ASensorManager_getInstance should be available
  assert(getInstanceFunc);
  dlclose(androidHandle);

  return getInstanceFunc();
}

static int pfd[2];

static void *thread_func()
{
    ssize_t rdsz;
    char buf[128];
    while((rdsz = read(pfd[0], buf, sizeof(buf) - 1)) > 0)
    {
        if(buf[rdsz - 1] == '\n') --rdsz;
        buf[rdsz] = 0; /* add null-terminator */
        LOGI("%s", buf);
    }

    close(pfd[0]);
    
    return 0;
}

/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(struct android_app* state) {
    LOGI("Entering android_main");

    /* make stdout line-buffered and stderr unbuffered */
    setvbuf(stdout, 0, _IOLBF, 0);
    setvbuf(stderr, 0, _IONBF, 0);

    /* create the pipe and redirect stdout and stderr */
    if(pipe(pfd) == -1)
    {
        LOGF("Unable to create pipe to redirect stdout/stderr to logs.");
        return;
    }

    dup2(pfd[1], 1);
    dup2(pfd[1], 2);

    /* spawn the logging thread */
    thread log_thread(thread_func);

    LOGI("STDOUT/STDERR > Log redirection thread started");

    state->userData = g_pApp;
    state->onAppCmd = engine_handle_cmd;
    state->onInputEvent = engine_handle_input;
    AndroidApplication* engine = dynamic_cast<AndroidApplication*>(g_pApp);
    engine->m_pApp = state;

    // Prepare to monitor accelerometer
    engine->m_pSensorManager = AcquireASensorManagerInstance(state);
    engine->m_pAccelerometerSensor = ASensorManager_getDefaultSensor(
                                        engine->m_pSensorManager,
                                        ASENSOR_TYPE_ACCELEROMETER);
    engine->m_pSensorEventQueue = ASensorManager_createEventQueue(
                                    engine->m_pSensorManager,
                                    state->looper, LOOPER_ID_USER,
                                    NULL, NULL);

    if (state->savedState != NULL) {
        // We are starting with a previous saved state; restore from it.
        engine->m_State = *(AndroidApplication::saved_state*)state->savedState;
    }

    // loop waiting for stuff to do.
    while (!g_pApp->IsQuit()) {
        // Read all pending events.
        int ident;
        int events;
        struct android_poll_source* source;

        // If not animating, we will block forever waiting for events.
        // If animating, we loop until all events are read, then continue
        // to draw the next frame of animation.
        while ((ident=ALooper_pollAll(engine->m_bAnimating ? 0 : -1, NULL, &events,
                                      (void**)&source)) >= 0) {

            // Process this event.
            if (source != NULL) {
                source->process(state, source);
            }

            // If a sensor has data, process it now.
            if (ident == LOOPER_ID_USER) {
                if (engine->m_pAccelerometerSensor != nullptr) {
                    ASensorEvent event;
                    while (ASensorEventQueue_getEvents(engine->m_pSensorEventQueue,
                                                       &event, 1) > 0) {
                        LOGV("accelerometer: x=%f y=%f z=%f",
                             event.acceleration.x, event.acceleration.y,
                             event.acceleration.z);
                    }
                }
            }

            // Check if we are exiting.
            if (state->destroyRequested != 0) {
                engine->Finalize();
                close(pfd[1]);
                log_thread.join();
                return;
            }
        }

        if (engine->m_bAnimating) {
            // Done with events; draw next animation frame.
            engine->m_State.angle += .01f;
            if (engine->m_State.angle > 1) {
                engine->m_State.angle = 0;
            }

            engine->Tick();
        }
    }
}
