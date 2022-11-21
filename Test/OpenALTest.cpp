#include <cstdio>
#include <cstdlib>

#include "AssetLoader.hpp"
#include "WAVE.hpp"

#include "AL/al.h"
#include "AL/alc.h"
#include "AL/alext.h"

#define NUM_BUFFERS 1
#define NUM_SOURCES 1
#define NUM_ENVIRONMENTS 1

ALenum err, format;

ALfloat listenerPos[] = { 0.0f, 0.0f, 4.0f };
ALfloat listenerVel[] = { 0.0f, 0.0f, 0.0f };
ALfloat listenerOri[] = { 0.0f, 0.0f, 1.0f,
                          0.0f, 1.0f, 0.0f };

ALfloat source0Pos[] = { -2.0f, 0.0f, 0.0f };
ALfloat source0Vel[] = {  0.0f, 0.0f, 0.0f };

ALuint buffer[NUM_BUFFERS];
ALuint source[NUM_SOURCES];
ALuint environment[NUM_ENVIRONMENTS];

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

/* InitAL opens a device and sets up a context using default attributes, making
 * the program ready to call OpenAL functions. */
int InitAL(char ***argv, int *argc)
{
    const ALCchar *name;
    ALCdevice *device;
    ALCcontext *ctx;

    /* Open and initialize a device */
    device = NULL;
    if(argc && argv && *argc > 1 && strcmp((*argv)[0], "-device") == 0)
    {
        device = alcOpenDevice((*argv)[1]);
        if(!device)
            fprintf(stderr, "Failed to open \"%s\", trying default\n", (*argv)[1]);
        (*argv) += 2;
        (*argc) -= 2;
    }
    if(!device)
        device = alcOpenDevice(NULL);
    if(!device)
    {
        fprintf(stderr, "Could not open a device!\n");
        return 1;
    }

    ctx = alcCreateContext(device, NULL);
    if(ctx == NULL || alcMakeContextCurrent(ctx) == ALC_FALSE)
    {
        if(ctx != NULL)
            alcDestroyContext(ctx);
        alcCloseDevice(device);
        fprintf(stderr, "Could not set a context!\n");
        return 1;
    }

    name = NULL;
    if(alcIsExtensionPresent(device, "ALC_ENUMERATE_ALL_EXT"))
        name = alcGetString(device, ALC_ALL_DEVICES_SPECIFIER);
    if(!name || alcGetError(device) != AL_NO_ERROR)
        name = alcGetString(device, ALC_DEVICE_SPECIFIER);
    printf("Opened \"%s\"\n", name);

    return 0;
}

/* CloseAL closes the device belonging to the current context, and destroys the
 * context. */
void CloseAL(void)
{
    ALCdevice *device;
    ALCcontext *ctx;

    ctx = alcGetCurrentContext();
    if(ctx == NULL)
        return;

    device = alcGetContextsDevice(ctx);

    alcMakeContextCurrent(NULL);
    alcDestroyContext(ctx);
    alcCloseDevice(device);
}


const char *FormatName(ALenum format)
{
    switch(format)
    {
    case AL_FORMAT_MONO8: return "Mono, U8";
    case AL_FORMAT_MONO16: return "Mono, S16";
    case AL_FORMAT_MONO_FLOAT32: return "Mono, Float32";
    case AL_FORMAT_STEREO8: return "Stereo, U8";
    case AL_FORMAT_STEREO16: return "Stereo, S16";
    case AL_FORMAT_STEREO_FLOAT32: return "Stereo, Float32";
    case AL_FORMAT_BFORMAT2D_8: return "B-Format 2D, U8";
    case AL_FORMAT_BFORMAT2D_16: return "B-Format 2D, S16";
    case AL_FORMAT_BFORMAT2D_FLOAT32: return "B-Format 2D, Float32";
    case AL_FORMAT_BFORMAT3D_8: return "B-Format 3D, U8";
    case AL_FORMAT_BFORMAT3D_16: return "B-Format 3D, S16";
    case AL_FORMAT_BFORMAT3D_FLOAT32: return "B-Format 3D, Float32";
    }
    return "Unknown Format";
}

void init(void)
{
    alListenerfv(AL_POSITION, listenerPos);
    alListenerfv(AL_VELOCITY, listenerVel);
    alListenerfv(AL_ORIENTATION, listenerOri);

    alGenBuffers(NUM_BUFFERS, buffer);

    err = alGetError();
    if(err != AL_NO_ERROR)
    {
        fprintf(stderr, "- Error creating buffers:%s\n", alGetString(err));
        exit(1);
    }

    My::AssetLoader assetLoader;
    My::WaveParser waveParser;
    auto audioFile = assetLoader.SyncOpenAndReadBinary("Audio/octave.wav");
    auto audioClip = waveParser.Parse(audioFile);

    ALenum format;

    getOpenALFormat(format, audioClip.format);

    alGenBuffers(1, &buffer[0]);
    alBufferData(buffer[0], format, audioClip.data, (ALsizei) audioClip.data_length, (ALsizei) audioClip.sample_rate);

    /* Check if an error occured, and clean up if so. */
    err = alGetError();
    if(err != AL_NO_ERROR)
    {
        fprintf(stderr, "OpenAL Error: %s\n", alGetString(err));
        if(alIsBuffer(buffer[0]))
            alDeleteBuffers(1, buffer);
        exit(1);
    }

    alGenSources(NUM_SOURCES, source);

    err = alGetError();
    if(err != AL_NO_ERROR)
    {
        fprintf(stderr, "- Error creating sources: %s\n", alGetString(err));
        exit(2);
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
    InitAL(&argv, &argc);

    init();

    alSourcePlay(source[0]);

    err = alGetError();
    if(err != AL_NO_ERROR)
    {
        fprintf(stderr, "- Error playback the sound: %s\n", alGetString(err));
        exit(3);
    }
 
    printf("press enter to exit.");

    char c;
    c = getchar();

    alSourceStop(source[0]);

    alDeleteSources(NUM_SOURCES, source);
    alDeleteBuffers(NUM_BUFFERS, buffer);

    CloseAL();

    return 0;
}
