#include <cstdio>
#include <cstdlib>
#include <AL/alut.h>

#define NUM_BUFFERS 1
#define NUM_SOURCES 1
#define NUM_ENVIRONMENTS 1

ALfloat listenerPos[] = { 0.0f, 0.0f, 4.0f };
ALfloat listenerVel[] = { 0.0f, 0.0f, 0.0f };
ALfloat listenerOri[] = { 0.0f, 0.0f, 1.0f,
                          0.0f, 1.0f, 0.0f };

ALfloat source0Pos[] = { -2.0f, 0.0f, 0.0f };
ALfloat source0Vel[] = {  0.0f, 0.0f, 0.0f };

ALuint buffer[NUM_BUFFERS];
ALuint source[NUM_SOURCES];
ALuint environemnt[NUM_ENVIRONMENTS];

void init(void)
{
    alListenerfv(AL_POSITION, listenerPos);
    alListenerfv(AL_VELOCITY, listenerVel);
    alListenerfv(AL_ORIENTATION, listenerOri);

    alGetError();

    alGenBuffers(NUM_BUFFERS, buffer);

    if(alGetError() != AL_NO_ERROR)
    {
        printf("- Error creating buffers !!\n");
        exit(1);
    }
    else
    {
        printf("init() - No errors yet.\n");
    }

    buffer[0] = alutCreateBufferFromFile("test.wave");

    alGetError();
    
    alGenSources(NUM_SOURCES, source);

    if(alGetError() != AL_NO_ERROR)
    {
        printf("- Error creating sources !!\n");
        exit(2);
    }
    else
    {
        printf("init() - no errors after alGenSources\n");
    }

    alSourcef(source[0], AL_PITCH, 1.0f);
    alSourcef(source[0], AL_GAIN, 1.0f);
    alSourcefv(source[0], AL_POSITION, source0Pos);
    alSourcefv(source[0], AL_VELOCITY, source0Vel);
    alSourcei(source[0], AL_BUFFER, buffer[0]);
    alSourcei(source[0], AL_LOOPING, AL_TRUE);
}

int main(int argc, char** argv)
{
    alutInit(&argc, argv);

    init();

    alGetError();
    alSourcePlay(source[0]);

    if(alGetError() != AL_NO_ERROR)
    {
        printf("- Error playback the sound!!\n");
        exit(3);
    }
    else
    {
        printf("no errors after alSourcePlay()\n");
    }
 
    printf("press enter to exit.");

    char c;
    c = getchar();

    return 0;
}
