/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if (defined(__linux__) || defined(__IPHONE_OS_VERSION_MAX_ALLOWED) || defined(__MAC_OS_X_VERSION_MAX_ALLOWED))

#include <pthread.h>

// Threads:
typedef pthread_t sample_platform_thread;

static inline int sample_platform_thread_create(sample_platform_thread *thread, void *(*func)(void *), void *data) {
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    return pthread_create(thread, &thread_attr, func, data);
}
static inline int sample_platform_thread_join(sample_platform_thread thread, void **retval) { return pthread_join(thread, retval); }

// Thread IDs:
typedef pthread_t sample_platform_thread_id;
static inline sample_platform_thread_id sample_platform_get_thread_id() { return pthread_self(); }

// Thread mutex:
typedef pthread_mutex_t sample_platform_thread_mutex;
static inline void sample_platform_thread_create_mutex(sample_platform_thread_mutex *pMutex) { pthread_mutex_init(pMutex, NULL); }
static inline void sample_platform_thread_lock_mutex(sample_platform_thread_mutex *pMutex) { pthread_mutex_lock(pMutex); }
static inline void sample_platform_thread_unlock_mutex(sample_platform_thread_mutex *pMutex) { pthread_mutex_unlock(pMutex); }
static inline void sample_platform_thread_delete_mutex(sample_platform_thread_mutex *pMutex) { pthread_mutex_destroy(pMutex); }
typedef pthread_cond_t sample_platform_thread_cond;
static inline void sample_platform_thread_init_cond(sample_platform_thread_cond *pCond) { pthread_cond_init(pCond, NULL); }
static inline void sample_platform_thread_cond_wait(sample_platform_thread_cond *pCond, sample_platform_thread_mutex *pMutex) {
    pthread_cond_wait(pCond, pMutex);
}
static inline void sample_platform_thread_cond_broadcast(sample_platform_thread_cond *pCond) { pthread_cond_broadcast(pCond); }

#elif defined(_WIN32)  // defined(__linux__)
/* Windows-specific common code: */
#include <winsock2.h>
#include <windows.h>

// Threads:
typedef HANDLE sample_platform_thread;
static inline int sample_platform_thread_create(sample_platform_thread *thread, void *(*func)(void *), void *data) {
    DWORD threadID;
    *thread = CreateThread(NULL,  // default security attributes
                           0,     // use default stack size
                           (LPTHREAD_START_ROUTINE)func,
                           data,        // thread function argument
                           0,           // use default creation flags
                           &threadID);  // returns thread identifier
    return (*thread != NULL);
}
static inline int sample_platform_thread_join(sample_platform_thread thread, void **retval) {
    return WaitForSingleObject(thread, INFINITE);
}

// Thread IDs:
typedef DWORD sample_platform_thread_id;
static sample_platform_thread_id sample_platform_get_thread_id() { return GetCurrentThreadId(); }

// Thread mutex:
typedef CRITICAL_SECTION sample_platform_thread_mutex;
static void sample_platform_thread_create_mutex(sample_platform_thread_mutex *pMutex) { InitializeCriticalSection(pMutex); }
static void sample_platform_thread_lock_mutex(sample_platform_thread_mutex *pMutex) { EnterCriticalSection(pMutex); }
static void sample_platform_thread_unlock_mutex(sample_platform_thread_mutex *pMutex) { LeaveCriticalSection(pMutex); }
static void sample_platform_thread_delete_mutex(sample_platform_thread_mutex *pMutex) { DeleteCriticalSection(pMutex); }
typedef CONDITION_VARIABLE sample_platform_thread_cond;
static void sample_platform_thread_init_cond(sample_platform_thread_cond *pCond) { InitializeConditionVariable(pCond); }
static void sample_platform_thread_cond_wait(sample_platform_thread_cond *pCond, sample_platform_thread_mutex *pMutex) {
    SleepConditionVariableCS(pCond, pMutex, INFINITE);
}
static void sample_platform_thread_cond_broadcast(sample_platform_thread_cond *pCond) { WakeAllConditionVariable(pCond); }
#else  // defined(_WIN32)

#error The "sample_common.h" file must be modified for this OS.

// NOTE: In order to support another OS, an #elif needs to be added (above the
// "#else // defined(_WIN32)") for that OS, and OS-specific versions of the
// contents of this file must be created.

// NOTE: Other OS-specific changes are also needed for this OS.  Search for
// files with "WIN32" in it, as a quick way to find files that must be changed.

#endif  // defined(_WIN32)
