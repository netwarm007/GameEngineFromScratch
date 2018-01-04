#include <unistd.h>
#include <thread>
#include <android/log.h>
#include "Logger.hpp"

using namespace std;

static int pfd[2];
static const char *tag = "MyGameEngine";

static void *thread_func()
{
    ssize_t rdsz;
    char buf[128];
    while((rdsz = read(pfd[0], buf, sizeof(buf) - 1)) > 0)
    {
        if(buf[rdsz - 1] == '\n') --rdsz;
        buf[rdsz] = 0; /* add null-terminator */
        __android_log_write(ANDROID_LOG_DEBUG, tag, buf);
    }
    
    return 0;
}

int start_logger(const char* app_name)
{
    tag = app_name;

    /* make stdout line-buffered and stderr unbuffered */
    setvbuf(stdout, 0, _IOLBF, 0);
    setvbuf(stderr, 0, _IONBF, 0);

    /* create the pipe and redirect stdout and stderr */
    pipe(pfd);
    dup2(pfd[1], 1);
    dup2(pfd[1], 2);

    /* spawn the logging thread */
    thread log_thread(thread_func);

    log_thread.join();

    return 0;
}

