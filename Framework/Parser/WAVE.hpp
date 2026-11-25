#pragma once
#include <cassert>
#include <cstdint>
#include "IAudioClipParser.hpp"

namespace My {
#pragma pack(push, 1)
struct WAVE_FILEHEADER {
    uint8_t Magic[4];
    uint32_t FileSize;
    uint8_t FileTypeHeader[4];
};

struct WAVE_CHUNKHEADER {
    uint8_t ChunkMarker[4];
    uint32_t ChunkSize;
};

struct WAVE_FORMAT_CHUNKHEADER : WAVE_CHUNKHEADER {
    uint16_t FormatType;
    uint16_t ChannelNum;
    uint32_t SampleRate;
    uint32_t ByteRate;
    uint16_t BlockSize;
    uint16_t BitsPerSample;
};

struct WAVE_DATA_CHUNKHEADER : WAVE_CHUNKHEADER {};
#pragma pack(pop)

constexpr uint32_t RIFF_HEADER_SIZE = offsetof(WAVE_FILEHEADER, FileTypeHeader);

class WaveParser : _implements_ AudioClipParser {
   public:
    AudioClip Parse(Buffer& buf) override {
        AudioClip audio_clip;

        const uint8_t* pData = buf.GetData();
        const uint8_t* pDataEnd = buf.GetData() + buf.GetDataSize();

        std::cerr << "Parsing as Wave file:" << std::endl;

        if (pData == nullptr ||
            buf.GetDataSize() < sizeof(WAVE_FILEHEADER)) {
            std::cerr << "Truncated WAVE header" << std::endl;
            return audio_clip;
        }

        const auto* pFileHeader =
            reinterpret_cast<const WAVE_FILEHEADER*>(pData);
        pData += sizeof(WAVE_FILEHEADER);

        if (!(pFileHeader->Magic[0] == 'R' && pFileHeader->Magic[1] == 'I' &&
              pFileHeader->Magic[2] == 'F' && pFileHeader->Magic[3] == 'F') ||
            !(pFileHeader->FileTypeHeader[0] == 'W' &&
              pFileHeader->FileTypeHeader[1] == 'A' &&
              pFileHeader->FileTypeHeader[2] == 'V' &&
              pFileHeader->FileTypeHeader[3] == 'E')) {
            std::cerr << "File is not a WAVE file!" << std::endl;
            return audio_clip;
        }

        bool format_found = false;
        while (pData < pDataEnd) {
            if (pDataEnd - pData < (ptrdiff_t)sizeof(WAVE_CHUNKHEADER)) {
                std::cerr << "Truncated WAVE chunk header" << std::endl;
                return audio_clip;
            }

            const auto* pChunkHeader =
                reinterpret_cast<const WAVE_CHUNKHEADER*>(pData);
            // chunk payload plus the 4-byte chunk header must fit in the file
            const size_t chunk_payload_size = pChunkHeader->ChunkSize;
            if (chunk_payload_size >
                (size_t)(pDataEnd - pData) - sizeof(WAVE_CHUNKHEADER)) {
                std::cerr << "Truncated WAVE chunk data" << std::endl;
                return audio_clip;
            }

            if (pChunkHeader->ChunkMarker[0] == 'f' &&
                pChunkHeader->ChunkMarker[1] == 'm' &&
                pChunkHeader->ChunkMarker[2] == 't' &&
                pChunkHeader->ChunkMarker[3] == ' ') {
                if (chunk_payload_size <
                    sizeof(WAVE_FORMAT_CHUNKHEADER) -
                        sizeof(WAVE_CHUNKHEADER)) {
                    std::cerr << "Truncated WAVE format chunk" << std::endl;
                    return audio_clip;
                }

                const auto* pFormatChunkHeader =
                    reinterpret_cast<const WAVE_FORMAT_CHUNKHEADER*>(pData);

                audio_clip.channel_num = pFormatChunkHeader->ChannelNum;
                audio_clip.sample_rate = pFormatChunkHeader->SampleRate;
                audio_clip.block_size = pFormatChunkHeader->BlockSize;
                audio_clip.bits_per_sample = pFormatChunkHeader->BitsPerSample;

                switch (audio_clip.channel_num) {
                    case 1:
                        switch (audio_clip.bits_per_sample) {
                            case 8:
                                audio_clip.format = AudioClipFormat::MONO_8;
                                break;

                            case 16:
                                audio_clip.format = AudioClipFormat::MONO_16;
                                break;

                            default:
                                std::cerr
                                    << "Unsupported WAVE sample format: mono "
                                    << (uint16_t)audio_clip.bits_per_sample
                                    << " bit" << std::endl;
                                return AudioClip {};
                        }
                        break;

                    case 2:
                        switch (audio_clip.bits_per_sample) {
                            case 8:
                                audio_clip.format = AudioClipFormat::STEREO_8;
                                break;

                            case 16:
                                audio_clip.format =
                                    AudioClipFormat::STEREO_16;
                                break;

                            default:
                                std::cerr
                                    << "Unsupported WAVE sample format: stereo "
                                    << (uint16_t)audio_clip.bits_per_sample
                                    << " bit" << std::endl;
                                return AudioClip {};
                        }
                        break;

                    default:
                        std::cerr << "Unsupported WAVE channel count: "
                                  << (uint16_t)audio_clip.channel_num
                                  << std::endl;
                        return AudioClip {};
                }
                format_found = true;
            } else if (pChunkHeader->ChunkMarker[0] == 'd' &&
                       pChunkHeader->ChunkMarker[1] == 'a' &&
                       pChunkHeader->ChunkMarker[2] == 't' &&
                       pChunkHeader->ChunkMarker[3] == 'a') {
                if (!format_found) {
                    std::cerr << "WAVE data chunk before format chunk"
                              << std::endl;
                    return AudioClip {};
                }
                audio_clip.data = pData + sizeof(WAVE_DATA_CHUNKHEADER);
                audio_clip.data_length = chunk_payload_size;
            } else {
                fprintf(
                    stderr, "Ignore Unknown Chunk: %c%c%c%c\n",
                    pChunkHeader->ChunkMarker[0], pChunkHeader->ChunkMarker[1],
                    pChunkHeader->ChunkMarker[2], pChunkHeader->ChunkMarker[3]);
            }

            pData += sizeof(WAVE_CHUNKHEADER);
            pData += chunk_payload_size;
            // RIFF chunks are word aligned
            if (chunk_payload_size & 1) pData += 1;
        }

        return audio_clip;
    }
};
}  // namespace My
