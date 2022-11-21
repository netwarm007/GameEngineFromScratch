#include <cstdio>
#include <cstdint>
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>

#include "WAVE.hpp"

const unsigned int SAMPLE_RATE = 11025U; // 11KHz
uint8_t buf[SAMPLE_RATE];

My::WAVE_FILEHEADER file_header = {
    {'R', 'I', 'F', 'F'},
    0,
    {'W', 'A', 'V', 'E'}
};

My::WAVE_FORMAT_CHUNKHEADER format_chunk_header = {
    {{'f', 'm', 't', ' '},
    sizeof(My::WAVE_FORMAT_CHUNKHEADER) - sizeof(My::WAVE_CHUNKHEADER)},
    1, // PCM
    1, // only 1 channel
    SAMPLE_RATE, // 11kHz
    1 * SAMPLE_RATE * 8 / 8, // byte rate
    1 * 8 / 8, // block align
    8 // bits per sample
};

My::WAVE_DATA_CHUNKHEADER data_chunk_header = {
    {{'d', 'a', 't', 'a'},
    0}
};

int main(int argc, char** argv) {
    // Create the wave file and open with binary write mode
    auto fp = std::fopen("noise.wav", "wb");

    if (fp) {
        // skip file header
        std::fseek(fp, sizeof(file_header), SEEK_SET);

        // write format chunk
        std::fwrite(&format_chunk_header, sizeof(format_chunk_header), 1, fp);

        // now generate the wave
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0, 255);
        for (int i = 0; i < SAMPLE_RATE; i++) {
            buf[i] = distribution(generator);  // generates number in the range 0..255
        }

        data_chunk_header.ChunkSize = SAMPLE_RATE;

        std::fwrite(&data_chunk_header, sizeof(data_chunk_header), 1, fp);
        std::fwrite(buf, sizeof(uint8_t), SAMPLE_RATE, fp);

        // Now fill the RIFF header with correct chunk size
        file_header.FileSize = std::ftell(fp) - My::RIFF_HEADER_SIZE;
        std::fseek(fp, 0, SEEK_SET);
        // write file header
        std::fwrite(&file_header, sizeof(file_header), 1, fp);

        // Now close the wave file
        std::fclose(fp);
    }

    // Create the wave file and open with binary write mode
    fp = std::fopen("sine.wav", "wb");

    if (fp) {
        // skip file header
        std::fseek(fp, sizeof(file_header), SEEK_SET);

        // write format chunk
        std::fwrite(&format_chunk_header, sizeof(format_chunk_header), 1, fp);

        // now generate the wave
        for (int i = 0; i < SAMPLE_RATE; i++) {
            buf[i] = 0xFF * std::sin(261.6f * 2.0f * M_PI * i / SAMPLE_RATE);
        }

        data_chunk_header.ChunkSize = SAMPLE_RATE;

        std::fwrite(&data_chunk_header, sizeof(data_chunk_header), 1, fp);
        std::fwrite(buf, sizeof(uint8_t), SAMPLE_RATE, fp);

        // Now fill the RIFF header with correct chunk size
        file_header.FileSize = std::ftell(fp) - My::RIFF_HEADER_SIZE;
        std::fseek(fp, 0, SEEK_SET);
        // write file header
        std::fwrite(&file_header, sizeof(file_header), 1, fp);

        // Now close the wave file
        std::fclose(fp);
    }

    // Create the wave file and open with binary write mode
    fp = std::fopen("octave.wav", "wb");

    if (fp) {
        // skip file header
        std::fseek(fp, sizeof(file_header), SEEK_SET);

        // write format chunk
        std::fwrite(&format_chunk_header, sizeof(format_chunk_header), 1, fp);

        data_chunk_header.ChunkSize = SAMPLE_RATE * 8;

        std::fwrite(&data_chunk_header, sizeof(data_chunk_header), 1, fp);

        // now generate the wave
        const float oct_frequency[] = {261.6f, 293.6f, 329.6f, 349.2f, 392.0f, 440.0f, 493.8f, 523.2f};
        for (int oct_index = 0; oct_index < 8; oct_index++) {
            for (int i = 0; i < SAMPLE_RATE; i++) {
                buf[i] = 0xFF * std::sin(oct_frequency[oct_index] * 2.0f * M_PI * i / SAMPLE_RATE);
            }
            std::fwrite(buf, sizeof(uint8_t), SAMPLE_RATE, fp);
        }

        // Now fill the RIFF header with correct chunk size
        file_header.FileSize = std::ftell(fp) - My::RIFF_HEADER_SIZE;
        std::fseek(fp, 0, SEEK_SET);
        // write file header
        std::fwrite(&file_header, sizeof(file_header), 1, fp);

        // Now close the wave file
        std::fclose(fp);
    }

    return 0;
}