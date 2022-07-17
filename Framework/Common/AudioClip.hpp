#pragma once
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace My {
    struct AudioClip {
        const void* data;
        size_t data_length;
        uint16_t channel_num;
        uint32_t sample_rate;
        uint32_t stride;
    };

    std::ostream& operator<<(std::ostream& out, const AudioClip& audio_clip);
}
