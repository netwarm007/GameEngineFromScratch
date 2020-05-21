#pragma once
#include <android/log.h>
#define LOGV(...)                                                   \
    ((void)__android_log_print(ANDROID_LOG_VERBOSE, "MyGameEngine", \
                               __VA_ARGS__))
#define LOGD(...) \
    ((void)__android_log_print(ANDROID_LOG_DEBUG, "MyGameEngine", __VA_ARGS__))
#define LOGI(...) \
    ((void)__android_log_print(ANDROID_LOG_INFO, "MyGameEngine", __VA_ARGS__))
#define LOGW(...) \
    ((void)__android_log_print(ANDROID_LOG_WARN, "MyGameEngine", __VA_ARGS__))
#define LOGE(...) \
    ((void)__android_log_print(ANDROID_LOG_ERROR, "MyGameEngine", __VA_ARGS__))
#define LOGF(...) \
    ((void)__android_log_print(ANDROID_LOG_FATAL, "MyGameEngine", __VA_ARGS__))
