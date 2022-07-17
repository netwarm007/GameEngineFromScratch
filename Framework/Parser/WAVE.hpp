#pragma onece
#include <cstdint>
#include "IAudioClipParser.hpp"

namespace My {
#pragma pack(push, 1)
struct WAVE_FILEHEADER {
    uint8_t Magic[4];
    uint32_t FileSize;
    uint8_t FileTypeHeader[4];
    uint8_t FormatChunkMarker[4];
    uint32_t FormatChunkSize;
    uint16_t FormatType;
    uint16_t ChannelNum;
    uint32_t SampleRate;
    uint32_t ByteRate;
    uint16_t Stride;
    uint16_t BitsPerSample;
    uint8_t DataChunkMarker[4];
    uint32_t DataChunkSize;
};
#pragma pack(pop)

class WaveParser : _implements_ AudioClipParser {
   public:
    AudioClip Parse(Buffer& buf) override {
        AudioClip audio_clip;

        const uint8_t* pData = buf.GetData();
        [[maybe_unused]] const uint8_t* pDataEnd =
            buf.GetData() + buf.GetDataSize();

        std::cerr << "Parsing as Wave file:" << std::endl;

        const auto* pFileHeader =
            reinterpret_cast<const WAVE_FILEHEADER*>(pData);
        pData += sizeof(WAVE_FILEHEADER);

        assert(pFileHeader->Magic[0] == 'R' && pFileHeader->Magic[1] == 'I' && pFileHeader->Magic[2] == 'F' && pFileHeader->Magic[3] == 'F'); 
        assert(pFileHeader->FileSize == buf.GetDataSize());
        assert(pFileHeader->FileTypeHeader[0] == 'W' && pFileHeader->FileTypeHeader[1] == 'A' && pFileHeader->FileTypeHeader[2] == 'V' && pFileHeader->FileTypeHeader[3] == 'E'); 
        assert(pFileHeader->FormatChunkMarker[0] == 'f' && pFileHeader->FormatChunkMarker[1] == 'm' && pFileHeader->FormatChunkMarker[2] == 't' && pFileHeader->FormatChunkMarker[3] == '\0'); 
        assert(pFileHeader->FormatChunkSize == 16);
        assert(pFileHeader->DataChunkMarker[0] == 'd' && pFileHeader->DataChunkMarker[1] == 'a' && pFileHeader->DataChunkMarker[2] == 't' && pFileHeader->DataChunkMarker[3] == 'a'); 
        assert(pData < pDataEnd);

        audio_clip.data = pData;
        audio_clip.data_length = pFileHeader->DataChunkSize;
        audio_clip.channel_num = pFileHeader->ChannelNum;
        audio_clip.sample_rate = pFileHeader->SampleRate;
        audio_clip.stride = pFileHeader->Stride;

        return audio_clip;
    }
};
}
