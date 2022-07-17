#include "AudioClip.hpp"

namespace My {
std::ostream& operator<<(std::ostream& out, const AudioClip& audio_clip) {
    out << "Audio Clip" << std::endl;
    out << "-----" << std::endl;
    out << "Channel Num: " << audio_clip.channel_num << std::endl;
    out << "Sample Rate: " << audio_clip.sample_rate << std::endl;
    out << "Stride: " << audio_clip.stride << std::endl;

    return out;
}
}
