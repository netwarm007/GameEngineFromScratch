#pragma once
#include <cstdio>
#include <iostream>
#include <string>
#include <cassert>
#include <queue>
#include <algorithm>
#include "ImageParser.hpp"
#include "portable.hpp"
#include "HuffmanTree.hpp"
#include "ColorSpaceConversion.hpp"

// Enable this to print out very detailed decode information
//#define DUMP_DETAILS

namespace My {
#pragma pack(push, 1)
    struct JFIF_FILEHEADER {
        uint16_t    SOI;
    };

    struct JPEG_SEGMENT_HEADER {
        uint16_t    Marker;
        uint16_t    Length;
    };

    struct APP0 : public JPEG_SEGMENT_HEADER {
        char        Identifier[5];
    };

    struct JFIF_APP0 : public APP0 {
        uint8_t     MajorVersion;
        uint8_t     MinorVersion;
        uint8_t     DensityUnits;
        uint16_t    Xdensity;
        uint16_t    Ydensity;
        uint8_t     Xthumbnail;
        uint8_t     Ythumbnail;
    };

    struct JFXX_APP0 : public APP0 {
        uint8_t     ThumbnailFormat;
    };

    struct FRAME_COMPONENT_SPEC_PARAMS {
        public:
            uint8_t     ComponentIdentifier;

        private:
            uint8_t     SamplingFactor;

        public:
            uint8_t     QuantizationTableDestSelector;

            uint16_t HorizontalSamplingFactor() const { return SamplingFactor >> 4; };
            uint16_t VerticalSamplingFactor() const { return SamplingFactor & 0x07; };
    };

    struct FRAME_HEADER : public JPEG_SEGMENT_HEADER {
        uint8_t     SamplePrecision;
        uint16_t    NumOfLines;
        uint16_t    NumOfSamplesPerLine;
        uint8_t     NumOfComponentsInFrame;
    };

    struct SCAN_COMPONENT_SPEC_PARAMS {
        public:
            uint8_t     ComponentSelector;

        private:
            uint8_t     EntropyCodingTableDestSelector;

        public:
            uint16_t     DcEntropyCodingTableDestSelector() const { return EntropyCodingTableDestSelector >> 4; };
            uint16_t     AcEntropyCodingTableDestSelector() const { return EntropyCodingTableDestSelector & 0x07; };
    };

    struct SCAN_HEADER : public JPEG_SEGMENT_HEADER {
        uint8_t     NumOfComponents;
    };

    struct QUANTIZATION_TABLE_SPEC {
        private:
            uint8_t data;

        public:
            uint16_t ElementPrecision() const { return data >> 4; };
            uint16_t DestinationIdentifier() const { return data & 0x07; };
    };

    struct HUFFMAN_TABLE_SPEC {
        private:
            uint8_t data;

        public:
            uint8_t NumOfHuffmanCodes[16];

            uint16_t TableClass() const { return data >> 4; };
            uint16_t DestinationIdentifier() const { return data & 0x07; };
    };

#pragma pack(pop)

    class JfifParser : implements ImageParser
    {
    private:
        const uint8_t m_zigzagIndex[64] = { 0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40, 33, 26, 19, 12, 5, 6, 13, 20, 27, 34, 41, 48, 56, 49, 42, 35, 28, 21, 14, 7,
                                    15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30, 23, 31, 38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63 };

    protected:
        HuffmanTree<uint8_t> m_treeHuffman[4];
        Matrix8X8i m_tableQuantization[4];
        std::vector<FRAME_COMPONENT_SPEC_PARAMS> m_tableFrameComponentsSpec;
        uint16_t m_nSamplePrecision;
        uint16_t m_nLines;
        uint16_t m_nSamplesPerLine;
        uint16_t m_nComponentsInFrame;

    public:
        virtual Image Parse(const Buffer& buf)
        {
            Image img;
            const uint8_t* pData = buf.GetData();
            const uint8_t* pDataEnd = buf.GetData() + buf.GetDataSize();

            const JFIF_FILEHEADER* pFileHeader = reinterpret_cast<const JFIF_FILEHEADER*>(pData);
            pData += sizeof(JFIF_FILEHEADER);
            if (pFileHeader->SOI == endian_net_unsigned_int((uint16_t)0xFFD8) /* FF D8 */) {
                std::cout << "Asset is JPEG file" << std::endl;

                while(pData < pDataEnd)
                {
                    bool foundStartOfScan = false;
                    size_t scanLength = 0;

                    const JPEG_SEGMENT_HEADER* pSegmentHeader = reinterpret_cast<const JPEG_SEGMENT_HEADER*>(pData);
                    std::cout << "============================" << std::endl;
                    std::printf("Segment Length: %d bytes\n" ,endian_net_unsigned_int(pSegmentHeader->Length));
                    switch (endian_net_unsigned_int(pSegmentHeader->Marker)) {
                        case 0xFFC0:
                        case 0xFFC2:
                            {
                                if (endian_net_unsigned_int(pSegmentHeader->Marker) == 0xFFC0)
                                    std::cout << "Start Of Frame0 (baseline DCT)" << std::endl;
                                else
                                    std::cout << "Start Of Frame2 (progressive DCT)" << std::endl;

                                std::cout << "----------------------------" << std::endl;

                                const FRAME_HEADER* pFrameHeader = reinterpret_cast<const FRAME_HEADER*>(pData);
                                m_nSamplePrecision = pFrameHeader->SamplePrecision;
                                m_nLines = endian_net_unsigned_int((uint16_t)pFrameHeader->NumOfLines);
                                m_nSamplesPerLine = endian_net_unsigned_int((uint16_t)pFrameHeader->NumOfSamplesPerLine);
                                m_nComponentsInFrame = pFrameHeader->NumOfComponentsInFrame;
                                
                                std::cout << "Sample Precision: " << m_nSamplePrecision << std::endl;
                                std::cout << "Num of Lines: " << m_nLines << std::endl;
                                std::cout << "Num of Samples per Line: " << m_nSamplesPerLine << std::endl;
                                std::cout << "Num of Components In Frame: " << m_nComponentsInFrame << std::endl;

                                const uint8_t* pTmp = pData + sizeof(FRAME_HEADER);
                                const FRAME_COMPONENT_SPEC_PARAMS* pFcsp = reinterpret_cast<const FRAME_COMPONENT_SPEC_PARAMS*>(pTmp);
                                for (uint8_t i = 0; i < pFrameHeader->NumOfComponentsInFrame; i++) {
                                    std::cout << "\tComponent Identifier: " << (uint16_t)pFcsp->ComponentIdentifier << std::endl;
                                    std::cout << "\tHorizontal Sampling Factor: " << (uint16_t)pFcsp->HorizontalSamplingFactor() << std::endl;
                                    std::cout << "\tVertical Sampling Factor: " << (uint16_t)pFcsp->VerticalSamplingFactor() << std::endl;
                                    std::cout << "\tQuantization Table Destination Selector: " << (uint16_t)pFcsp->QuantizationTableDestSelector << std::endl;
                                    std::cout << std::endl;
                                    m_tableFrameComponentsSpec.push_back(*pFcsp);
                                    pFcsp++;
                                } 

                                img.Width = m_nSamplesPerLine;
                                img.Height = m_nLines;
                                img.bitcount = 24;
                                img.pitch = ((img.Width * img.bitcount >> 3) + 3) & ~3;
                                img.data_size = img.pitch * img.Height;
                                img.data = g_pMemoryManager->Allocate(img.data_size);
                            }
                            break;
                        case 0xFFC4:
                            {
                                std::cout << "Define Huffman Table" << std::endl;
                                std::cout << "----------------------------" << std::endl;

                                auto segmentLength = endian_net_unsigned_int(pSegmentHeader->Length) - 2;

                                const uint8_t* pTmp = pData + sizeof(JPEG_SEGMENT_HEADER);

                                while (segmentLength > 0) {
                                    const HUFFMAN_TABLE_SPEC* pHtable = reinterpret_cast<const HUFFMAN_TABLE_SPEC*>(pTmp);
                                    std::cout << "Table Class: " << pHtable->TableClass() << std::endl;
                                    std::cout << "Destination Identifier: " << pHtable->DestinationIdentifier() << std::endl;

                                    const uint8_t* pCodeValueStart = reinterpret_cast<const uint8_t*>(pHtable) + sizeof(HUFFMAN_TABLE_SPEC);

                                    auto num_symbo = m_treeHuffman[(pHtable->TableClass() << 1) | pHtable->DestinationIdentifier()].PopulateWithHuffmanTable(pHtable->NumOfHuffmanCodes, pCodeValueStart);

#ifdef DUMP_DETAILS
                                    m_treeHuffman[(pHtable->TableClass() << 1) | pHtable->DestinationIdentifier()].Dump();
#endif

                                    auto processed_length = sizeof(HUFFMAN_TABLE_SPEC) + num_symbo;
                                    pTmp += processed_length;
                                    segmentLength -= processed_length;
                                }
                            }
                            break;
                        case 0xFFDB:
                            {
                                std::cout << "Define Quantization Table" << std::endl;
                                std::cout << "----------------------------" << std::endl;

                                auto segmentLength = endian_net_unsigned_int(pSegmentHeader->Length) - 2;

                                const uint8_t* pTmp = pData + sizeof(JPEG_SEGMENT_HEADER);

                                while (segmentLength > 0) {
                                    const QUANTIZATION_TABLE_SPEC* pQtable = reinterpret_cast<const QUANTIZATION_TABLE_SPEC*>(pTmp);
                                    std::cout << "Element Precision: " << pQtable->ElementPrecision() << std::endl;
                                    std::cout << "Destination Identifier: " << pQtable->DestinationIdentifier() << std::endl;

                                    const uint8_t* pElementDataStart = reinterpret_cast<const uint8_t*>(pQtable) + sizeof(QUANTIZATION_TABLE_SPEC);

                                    for (int i = 0; i < 64; i++) {
                                        int index = m_zigzagIndex[i];
                                        if (pQtable->ElementPrecision() == 0) {
                                            m_tableQuantization[pQtable->DestinationIdentifier()][index >> 3][index & 0x7] = pElementDataStart[i];
                                        } else {
                                            m_tableQuantization[pQtable->DestinationIdentifier()][index >> 3][index & 0x7] = endian_net_unsigned_int(*((uint16_t*)pElementDataStart + i));
                                        }
                                    }
#ifdef DUMP_DETAILS
                                    std::cout << m_tableQuantization[pQtable->DestinationIdentifier()];
#endif

                                    auto processed_length = sizeof(QUANTIZATION_TABLE_SPEC) + 64 * (pQtable->ElementPrecision() + 1);
                                    pTmp += processed_length;
                                    segmentLength -= processed_length;
                                }
                            }
                            break;
                        case 0xFFDD:
                            {
                                std::cout << "Define Restart Interval" << std::endl;
                                std::cout << "----------------------------" << std::endl;
                            }
                            break;
                        case 0xFFDA:
                            {
                                foundStartOfScan = true;
                                std::cout << "Start Of Scan" << std::endl;
                                std::cout << "----------------------------" << std::endl;

                                SCAN_HEADER* pScanHeader = (SCAN_HEADER*) pData;
                                std::cout << "Image Conponents in Scan: " << (uint16_t)pScanHeader->NumOfComponents << std::endl;

                                std::vector<uint8_t> scan_data;

                                {
                                    const uint8_t* pImageData = pData + endian_net_unsigned_int((uint16_t)pScanHeader->Length) + 2;

                                    // scan for scan data buffer size and remove bitstuff
                                    bool bitstuff = false;
                                    while (pImageData < pDataEnd && (*pImageData != 0xFF || *(pImageData + 1) == 0x00)) {
                                        if(!bitstuff) {
                                            scan_data.push_back(*pImageData);
                                        } else {
                                            // ignore it and reset the flag
                                            bitstuff = false;
                                        }

                                        if(*pImageData == 0xFF) {
                                            bitstuff = true;
                                        }

                                        pImageData++;
                                        scanLength++;
                                    }

                                    if (*(uint16_t*)pImageData == endian_net_unsigned_int((uint16_t)0xFFD9)) {
                                        std::cout << "Size Of Scan: " << scanLength << " bytes" << std::endl;
                                        std::cout << "Size Of Scan (after remove bitstuff): " << scan_data.size() << " bytes" << std::endl;
                                    } else {
                                        std::cout << "Find EOF when searching for EOS" << std::endl;
                                    }
                                }
                                
                                size_t byte_offset = 0;
                                uint8_t bit_offset = 0;

                                const uint8_t* pTmp = pData + sizeof(SCAN_HEADER);
                                const SCAN_COMPONENT_SPEC_PARAMS* pScsp = reinterpret_cast<const SCAN_COMPONENT_SPEC_PARAMS*>(pTmp);
                                int mcu_index = 0;
                                int mcu_count_x = ((m_nSamplesPerLine + 7) >> 3);
                                int mcu_count_y = ((m_nLines + 7) >> 3);
                                int mcu_count = mcu_count_x * mcu_count_y;
                                int16_t previous_dc[m_nComponentsInFrame];
                                memset(previous_dc, 0x00, sizeof(previous_dc));

                                std::cout << "Total MCU count: " << mcu_count << std::endl;
                                
                                while (byte_offset < scan_data.size() && mcu_index < mcu_count) {
#if DUMP_DETAILS
                                    std::cout << "MCU: " << mcu_index << std::endl;
#endif
                                    Matrix8X8i block[m_nComponentsInFrame];
                                    memset(&block, 0x00, sizeof(block));

                                    for (uint8_t i = 0; i < pScanHeader->NumOfComponents; i++) {
                                        const FRAME_COMPONENT_SPEC_PARAMS& fcsp = m_tableFrameComponentsSpec[i];
#if DUMP_DETAILS
                                        std::cout << "\tComponent Selector: " << (uint16_t)pScsp[i].ComponentSelector << std::endl;
                                        std::cout << "\tQuantization Table Destination Selector: " << (uint16_t)fcsp.QuantizationTableDestSelector << std::endl;
                                        std::cout << "\tDC Entropy Coding Table Destination Selector: " << (uint16_t)pScsp[i].DcEntropyCodingTableDestSelector() << std::endl;
                                        std::cout << "\tAC Entropy Coding Table Destination Selector: " << (uint16_t)pScsp[i].AcEntropyCodingTableDestSelector() << std::endl;
#endif

                                        // Decode DC
                                        uint8_t dc_code = m_treeHuffman[pScsp[i].DcEntropyCodingTableDestSelector()].DecodeSingleValue(scan_data.data(), scan_data.size(), &byte_offset, &bit_offset);
                                        uint8_t dc_bit_length = dc_code & 0x0F;
                                        int16_t dc_value;
                                        uint32_t tmp_value;

                                        if (!dc_code)
                                        {
#if DUMP_DETAILS
                                            std::cout << "Found EOB when decode DC!" << std::endl;
#endif
                                            dc_value = 0;
                                        } else {
                                            if (dc_bit_length + bit_offset <= 8)
                                            {
                                                tmp_value = ((scan_data[byte_offset] << bit_offset) & 0x00FF) >> bit_offset >> (8 - dc_bit_length - bit_offset);
                                            }
                                            else
                                            {
                                                tmp_value = scan_data[byte_offset] << 24 | scan_data[byte_offset+1] << 16 | scan_data[byte_offset+2] << 8 | scan_data[byte_offset+3];
                                                tmp_value = ((uint32_t)(tmp_value << bit_offset)) >> bit_offset >> (32 - dc_bit_length - bit_offset);
                                            }

                                            // decode dc value
                                            if ((tmp_value >> (dc_bit_length - 1)) == 0) {
                                                // MSB = 1, turn it to minus value
                                                dc_value = -(~tmp_value & ((0x0001 << dc_bit_length) - 1));
                                            } else {
                                                dc_value = tmp_value;
                                            }
                                        }

                                        // add with previous DC value
                                        dc_value += previous_dc[i];
                                        // save the value for next DC
                                        previous_dc[i] = dc_value;

#ifdef DUMP_DETAILS
                                        printf("DC Code: %x\n", dc_code);
                                        printf("DC Bit Length: %d\n", dc_bit_length);
                                        printf("DC Value: %d\n", dc_value);
#endif

                                        block[i][0][0] = dc_value;

                                        // forward pointers to end of DC
                                        bit_offset += dc_bit_length;
                                        while (bit_offset >= 8 ) {
                                            bit_offset -= 8;
                                            byte_offset++;
                                        }

                                        // Decode AC 
                                        int ac_index = 0;
                                        while (byte_offset < scan_data.size() && ac_index < 64)
                                        {
                                            uint8_t ac_code = m_treeHuffman[2 + pScsp[i].AcEntropyCodingTableDestSelector()].DecodeSingleValue(scan_data.data(), scan_data.size(), &byte_offset, &bit_offset);

                                            if (!ac_code)
                                            {
#if DUMP_DETAILS
                                                std::cout << "Found EOB when decode AC!" << std::endl;
#endif
                                                break;
                                            }
                                            else if (ac_code == 0xF0)
                                            {
#if DUMP_DETAILS
                                                std::cout << "Found ZRL when decode AC!" << std::endl;
#endif
                                                ac_index += 15;
                                                break;
                                            }

                                            uint8_t ac_zero_length = ac_code >> 4;
                                            ac_index += ac_zero_length;
                                            uint8_t ac_bit_length = ac_code & 0x0F;
                                            if (ac_bit_length + bit_offset <= 8)
                                            {
                                                tmp_value = ((scan_data[byte_offset] << bit_offset) & 0x00FF) >> bit_offset >> (8 - ac_bit_length - bit_offset);
                                            }
                                            else
                                            {
                                                tmp_value = scan_data[byte_offset] << 24 | scan_data[byte_offset+1] << 16 | scan_data[byte_offset+2] << 8 | scan_data[byte_offset+3];
                                                tmp_value = ((uint32_t)(tmp_value << bit_offset)) >> bit_offset >> (32 - ac_bit_length - bit_offset);
                                            }

                                            int16_t ac_value = (int16_t) tmp_value;

#ifdef DUMP_DETAILS
                                            printf("AC Code: %x\n", ac_code);
                                            printf("AC Bit Length: %d\n", ac_bit_length);
                                            printf("AC Value: %d\n", ac_value);
#endif

                                            int index = m_zigzagIndex[ac_index];
                                            block[i][index >> 3][index & 0x07] = ac_value;

                                            // forward pointers to end of AC
                                            bit_offset += ac_bit_length;
                                            while (bit_offset >= 8 ) {
                                                bit_offset -= 8;
                                                byte_offset++;
                                            }

                                            ac_index++;
                                        }

#ifdef DUMP_DETAILS
                                        printf("Extracted Component[%d] 8x8 block: ", i);
                                        std::cout << block[i];
                                        MatrixMulByElementi32(block[i], block[i], m_tableQuantization[fcsp.QuantizationTableDestSelector]);
                                        std::cout << "After Quantization: " << block[i];
                                        block[i] = IDCT8X8(block[i]);
                                        std::cout << "After IDCT: " << block[i];
#endif
                                    } 

                                    assert(m_nComponentsInFrame < 4);

                                    YCbCru8 ycbcr;
                                    auto mcu_index_x = mcu_index % mcu_count_x;
                                    auto mcu_index_y = mcu_index / mcu_count_x;
                                    uint8_t* pBuf;

                                    for (int i = 0; i < 8; i++) {
                                        for (int j = 0; j < 8; j++) {
                                            for (int k = 0; k < m_nComponentsInFrame; k++) {
                                                ycbcr[k] = std::clamp(block[k][i][j] + 128, 0, 255);
                                            }

                                            pBuf = reinterpret_cast<uint8_t*>(img.data)
                                                + (img.pitch * (mcu_index_y * 8 + i) + (mcu_index_x * 8 + j) * sizeof(RGBu8));
                                            reinterpret_cast<RGBu8*>(pBuf)->rgb = ConvertYCbCr2RGB(ycbcr);
                                        }
                                    }

                                    mcu_index++;
                                }
                            }
                            break;
                        case 0xFFD9:
                            {
                                std::cout << "End Of Scan" << std::endl;
                                std::cout << "----------------------------" << std::endl;
                            }
                            break;
                        case 0xFFE0:
                            {
                                const APP0* pApp0 = reinterpret_cast<const APP0*>(pData);
                                switch (endian_net_unsigned_int(*(uint32_t*)pApp0->Identifier)) {
                                    case "JFIF\0"_u32: 
                                        {
                                            const JFIF_APP0* pJfifApp0 = reinterpret_cast<const JFIF_APP0*>(pApp0);
                                            std::cout << "JFIF-APP0" << std::endl;
                                            std::cout << "----------------------------" << std::endl;
                                            std::cout << "JFIF Version: " << (uint16_t)pJfifApp0->MajorVersion << "." 
                                                << (uint16_t)pJfifApp0->MinorVersion << std::endl;
                                            std::cout << "Density Units: " << 
                                                ((pJfifApp0->DensityUnits == 0)?"No units" : 
                                                 ((pJfifApp0->DensityUnits == 1)?"Pixels per inch" : "Pixels per centimeter" )) 
                                                << std::endl;
                                            std::cout << "Density: " << endian_net_unsigned_int(pJfifApp0->Xdensity) << "*" 
                                                << endian_net_unsigned_int(pJfifApp0->Ydensity) << std::endl;
                                            if (pJfifApp0->Xthumbnail && pJfifApp0->Ythumbnail) {
                                                std::cout << "Thumbnail Dimesions [w*h]: " << (uint16_t)pJfifApp0->Xthumbnail << "*" 
                                                    << (uint16_t)pJfifApp0->Ythumbnail << std::endl;
                                            } else {
                                                std::cout << "No thumbnail defined in JFIF-APP0 segment!" << std::endl;
                                            }
                                        }
                                        break;
                                    case "JFXX\0"_u32:
                                        {
                                            const JFXX_APP0* pJfxxApp0 = reinterpret_cast<const JFXX_APP0*>(pApp0);
                                            std::cout << "Thumbnail Format: ";
                                            switch (pJfxxApp0->ThumbnailFormat) {
                                                case 0x10:
                                                    std::cout << "JPEG format";
                                                    break;
                                                case 0x11:
                                                    std::cout << "1 byte per pixel palettized format";
                                                    break;
                                                case 0x13:
                                                    std::cout << "3 byte per pixel RGB format";
                                                    break;
                                                default:
                                                    std::printf("Unrecognized Thumbnail Format: %x\n", pJfxxApp0->ThumbnailFormat);
                                            }
                                            std::cout << std::endl;
                                        }
                                    default:
                                        std::cout << "Ignor Unrecognized APP0 segment." << std::endl;
                                }
                            }
                            break;
                        case 0xFFFE:
                            {
                                std::cout << "Text Comment" << std::endl;
                                std::cout << "----------------------------" << std::endl;
                            }
                            break;
                        default:
                            std::printf("Ignor Unrecognized Segment. Marker=%0x\n", endian_net_unsigned_int(pSegmentHeader->Marker));
                            break;
                    }

                    pData += endian_net_unsigned_int(pSegmentHeader->Length) + 2 /* length of marker */;

                    if (foundStartOfScan) {
                        // jump to the end of the scan data
                        pData += scanLength;
                        foundStartOfScan = false;
                    }
                }
            }
            else {
                std::cout << "File is not a JPEG file!" << std::endl;
            }

            return img;
        }
    };
}



