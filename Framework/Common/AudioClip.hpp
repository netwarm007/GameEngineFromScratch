#pragma once
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace My {
    enum class AudioClipFormat : uint16_t {
        MONO_8,
        MONO_16,
        STEREO_8,
        STEREO_16
    };

    struct AudioClip {
        const void* data;
        size_t data_length;
        uint16_t channel_num;
        uint16_t bits_per_sample;
        uint32_t sample_rate;
        uint32_t block_size;
        AudioClipFormat format;
    };

    std::ostream& operator<<(std::ostream& out, const AudioClip& audio_clip);
}
