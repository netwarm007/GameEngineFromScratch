#pragma once
#include <cstdio>
#include <iostream>
#include <string>
#include <cassert>
#include <queue>
#include <algorithm>
#include "ImageParser.hpp"
#include "portable.hpp"

// Enable this to print out very detailed decode information
//#define DUMP_DETAILS 1

namespace My {
#pragma pack(push, 1)
    struct PNG_FILEHEADER {
        uint64_t Signature;
    };

    ENUM(PNG_CHUNK_TYPE) {
        IHDR = "IHDR"_u32,
        PLTE = "PLTE"_u32,
        IDAT = "IDAT"_u32,
        IEND = "IEND"_u32
    };

    static std::ostream& operator<<(std::ostream& out, PNG_CHUNK_TYPE type)
    {
        int32_t n = static_cast<int32_t>(type);
        n = endian_net_unsigned_int<int32_t>(n);
        char* c = reinterpret_cast<char*>(&n);
         
        for (size_t i = 0; i < sizeof(int32_t); i++) {
            out << *c++;
        }

        return out;
    }

    struct PNG_CHUNK_HEADER {
        uint32_t        Length;
        PNG_CHUNK_TYPE  Type;
    };
#pragma pack(pop)

    class PngParser : implements ImageParser
    {
    private:

    protected:

    public:
        virtual Image Parse(const Buffer& buf)
        {
            Image img;

            const uint8_t* pData = buf.GetData();
            const uint8_t* pDataEnd = buf.GetData() + buf.GetDataSize();

            const PNG_FILEHEADER* pFileHeader = reinterpret_cast<const PNG_FILEHEADER*>(pData);
            pData += sizeof(PNG_FILEHEADER);
            if (pFileHeader->Signature == endian_net_unsigned_int((uint64_t)0x89504E470D0A1A0A)) {
                std::cout << "Asset is PNG file" << std::endl;

                while(pData < pDataEnd)
                {
                    const PNG_CHUNK_HEADER * pChunkHeader = reinterpret_cast<const PNG_CHUNK_HEADER*>(pData);
                    PNG_CHUNK_TYPE type = static_cast<PNG_CHUNK_TYPE>(endian_net_unsigned_int(static_cast<uint32_t>(pChunkHeader->Type)));

#if DUMP_DETAILS
                    std::cout << "============================" << std::endl;
#endif
                    switch (type) 
                    {
                        case PNG_CHUNK_TYPE::IHDR:
                            {
                                std::cout << "IHDR (Image Header)" << std::endl;
                                std::cout << "----------------------------" << std::endl;
                            }
                            break;
                        case PNG_CHUNK_TYPE::PLTE:
                            {
                                std::cout << "PLTE (Palette)" << std::endl;
                                std::cout << "----------------------------" << std::endl;
                            }
                            break;
                        case PNG_CHUNK_TYPE::IDAT:
                            {
                                std::cout << "IDAT (Image Data Start)" << std::endl;
                                std::cout << "----------------------------" << std::endl;
                            }
                            break;
                        case PNG_CHUNK_TYPE::IEND:
                            {
                                std::cout << "IEND (Image Data End)" << std::endl;
                                std::cout << "----------------------------" << std::endl;
                            }
                            break;
                        default:
                            {
                                std::cout << "Ignor Unrecognized Chunk. Marker=" << type << std::endl;
                            }
                            break;
                    }
                    pData += endian_net_unsigned_int(pChunkHeader->Length) + sizeof(PNG_CHUNK_HEADER) + 4/* length of CRC */;
                }
            }
            else {
                std::cout << "File is not a PNG file!" << std::endl;
            }

            return img;
        }
    };
}



