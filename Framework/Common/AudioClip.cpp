#include "AudioClip.hpp"

namespace My {
std::ostream& operator<<(std::ostream& out, const AudioClip& audio_clip) {
    out << "Audio Clip" << std::endl;
    out << "-----" << std::endl;
    out << "Channel Num: " << audio_clip.channel_num << std::endl;
    out << "Sample Rate: " << audio_clip.sample_rate << std::endl;
    out << "Block Size: " << audio_clip.block_size << std::endl;

    return out;
}
}
