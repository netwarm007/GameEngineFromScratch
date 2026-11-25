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
        const void* data{nullptr};
        size_t data_length{0};
        uint16_t channel_num{0};
        uint16_t bits_per_sample{0};
        uint32_t sample_rate{0};
        uint32_t block_size{0};
        AudioClipFormat format{AudioClipFormat::MONO_8};
    };

    std::ostream& operator<<(std::ostream& out, const AudioClip& audio_clip);
}
