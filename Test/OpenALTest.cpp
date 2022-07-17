#include <cstdio>
#include <cstdlib>

#define AL_LIBTYPE_STATIC
#include "AL/al.h"
#include "AL/alc.h"

#include "AssetLoader.hpp"
#include "WAVE.hpp"

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

static void getOpenALFormat(ALenum& out_format, My::AudioClipFormat in_format) {
    switch(in_format) {
        case My::AudioClipFormat::MONO_8:
            out_format = AL_FORMAT_MONO8;
            break;
        case My::AudioClipFormat::MONO_16:
            out_format = AL_FORMAT_MONO16;
            break;
        case My::AudioClipFormat::STEREO_8:
            out_format = AL_FORMAT_STEREO8;
            break;
        case My::AudioClipFormat::STEREO_16:
            out_format = AL_FORMAT_STEREO16;
            break;
        default:
            assert(0);
    }
}

static void test(const char* filename)
{
    alListenerfv(AL_POSITION, listenerPos);
    alListenerfv(AL_VELOCITY, listenerVel);
    alListenerfv(AL_ORIENTATION, listenerOri);

    alGenBuffers(NUM_BUFFERS, buffer);

    if(alGetError() != AL_NO_ERROR)
    {
        printf("- Error creating buffers !!\n");
        exit(1);
    }

    My::AssetLoader assetLoader;
    My::WaveParser waveParser;
    auto audioData = assetLoader.SyncOpenAndReadBinary(filename);
    auto audioClip = waveParser.Parse(audioData);

    ALenum format;

    getOpenALFormat(format, audioClip.format);

    alBufferData(buffer[0], format, audioClip.data, (ALsizei) audioClip.data_length, (ALsizei) audioClip.sample_rate);

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
    ALCdevice *device;
    ALCcontext *context;

    device = alcOpenDevice(NULL);
    assert(device);

    context = alcCreateContext(device, NULL);
    assert(context);

    if (!alcMakeContextCurrent(context))
    {
        alcDestroyContext(context);
        alcCloseDevice(device);
        assert(0);

        return -1;
    }

    if (argc > 1) {
        test(argv[1]);
    } else {
        test("Audio/test.wav");
    }

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

    alDeleteSources(NUM_SOURCES, source);
    alDeleteBuffers(NUM_BUFFERS, buffer);

    alcDestroyContext(context);
    alcCloseDevice(device);

    return 0;
}
