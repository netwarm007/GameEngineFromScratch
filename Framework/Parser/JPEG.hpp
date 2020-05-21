#pragma once
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <queue>
#include <string>

#include "ColorSpaceConversion.hpp"
#include "HuffmanTree.hpp"
#include "ImageParser.hpp"
#include "portable.hpp"

// Enable this to print out very detailed decode information
//#define DUMP_DETAILS 1

namespace My {
#pragma pack(push, 1)
struct JFIF_FILEHEADER {
    uint16_t SOI;
};

struct JPEG_SEGMENT_HEADER {
    uint16_t Marker;
    uint16_t Length;
};

struct APP0 : public JPEG_SEGMENT_HEADER {
    char Identifier[5];
};

struct JFIF_APP0 : public APP0 {
    uint8_t MajorVersion;
    uint8_t MinorVersion;
    uint8_t DensityUnits;
    uint16_t Xdensity;
    uint16_t Ydensity;
    uint8_t Xthumbnail;
    uint8_t Ythumbnail;
};

struct JFXX_APP0 : public APP0 {
    uint8_t ThumbnailFormat;
};

struct FRAME_COMPONENT_SPEC_PARAMS {
   public:
    uint8_t ComponentIdentifier;

   private:
    uint8_t SamplingFactor;

   public:
    uint8_t QuantizationTableDestSelector;

    [[nodiscard]] uint16_t HorizontalSamplingFactor() const {
        return SamplingFactor >> 4;
    };
    [[nodiscard]] uint16_t VerticalSamplingFactor() const {
        return SamplingFactor & 0x07;
    };
};

struct FRAME_HEADER : public JPEG_SEGMENT_HEADER {
    uint8_t SamplePrecision;
    uint16_t NumOfLines;
    uint16_t NumOfSamplesPerLine;
    uint8_t NumOfComponentsInFrame;
};

struct SCAN_COMPONENT_SPEC_PARAMS {
   public:
    uint8_t ComponentSelector;

   private:
    uint8_t EntropyCodingTableDestSelector;

   public:
    [[nodiscard]] uint16_t DcEntropyCodingTableDestSelector() const {
        return EntropyCodingTableDestSelector >> 4;
    };
    [[nodiscard]] uint16_t AcEntropyCodingTableDestSelector() const {
        return EntropyCodingTableDestSelector & 0x07;
    };
};

struct SCAN_HEADER : public JPEG_SEGMENT_HEADER {
    uint8_t NumOfComponents;
};

struct QUANTIZATION_TABLE_SPEC {
   private:
    uint8_t data;

   public:
    [[nodiscard]] uint16_t ElementPrecision() const { return data >> 4; };
    [[nodiscard]] uint16_t DestinationIdentifier() const {
        return data & 0x07;
    };
};

struct HUFFMAN_TABLE_SPEC {
   private:
    uint8_t data;

   public:
    uint8_t NumOfHuffmanCodes[16];

    [[nodiscard]] uint16_t TableClass() const { return data >> 4; };
    [[nodiscard]] uint16_t DestinationIdentifier() const {
        return data & 0x07;
    };
};

struct RESTART_INTERVAL_DEF : public JPEG_SEGMENT_HEADER {
    uint16_t RestartInterval;
};

#pragma pack(pop)

class JfifParser : _implements_ ImageParser {
   private:
    const uint8_t m_zigzagIndex[64] = {
        0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

   protected:
    HuffmanTree<uint8_t> m_treeHuffman[4];
    Matrix8X8f m_tableQuantization[4];
    std::vector<FRAME_COMPONENT_SPEC_PARAMS> m_tableFrameComponentsSpec;
    uint16_t m_nSamplePrecision;
    uint16_t m_nLines;
    uint16_t m_nSamplesPerLine;
    uint16_t m_nComponentsInFrame;
    uint16_t m_nRestartInterval = 0;
    int mcu_index;
    int mcu_count_x;
    int mcu_count_y;
    int mcu_count;
    const SCAN_COMPONENT_SPEC_PARAMS* pScsp;

   protected:
    size_t parseScanData(const uint8_t* pScanData, const uint8_t* pDataEnd,
                         Image& img) {
        std::vector<uint8_t> scan_data;
        size_t scanLength = 0;

        {
            const uint8_t* p = pScanData;

            // scan for scan data buffer size and remove bitstuff
            bool bitstuff = false;
            while (p < pDataEnd && (*p != 0xFF || *(p + 1) == 0x00)) {
                if (!bitstuff) {
                    scan_data.push_back(*p);
                } else {
                    // ignore it and reset the flag
                    assert(*p == 0x00);
                    bitstuff = false;
                }

                if (*(uint16_t*)p ==
                    endian_net_unsigned_int((uint16_t)0xFF00)) {
                    bitstuff = true;
                }

                p++;
                scanLength++;
            }

            if (*p == 0xFF && *(p + 1) >= 0xD0 && *(p + 1) <= 0xD7) {
                // found restart mark
#if DUMP_DETAILS
                std::cerr << "Found RST while scan the ECS." << std::endl;
#endif
            }

#if DUMP_DETAILS
            std::cerr << "Size Of Scan: " << scanLength << " bytes"
                      << std::endl;
            std::cerr << "Size Of Scan (after remove bitstuff): "
                      << scan_data.size() << " bytes" << std::endl;
#endif
        }

        int16_t
            previous_dc[4];  // 4 is max num of components defined by ITU-T81
        memset(previous_dc, 0x00, sizeof(previous_dc));

        size_t byte_offset = 0;
        uint8_t bit_offset = 0;

        while (byte_offset < scan_data.size() && mcu_index < mcu_count) {
#if DUMP_DETAILS
            std::cerr << "MCU: " << mcu_index << std::endl;
#endif
            Matrix8X8f
                block[4];  // 4 is max num of components defined by ITU-T81
            memset(&block, 0x00, sizeof(block));

            for (uint8_t i = 0; i < m_nComponentsInFrame; i++) {
                const FRAME_COMPONENT_SPEC_PARAMS& fcsp =
                    m_tableFrameComponentsSpec[i];
#if DUMP_DETAILS
                std::cerr << "\tComponent Selector: "
                          << (uint16_t)pScsp[i].ComponentSelector << std::endl;
                std::cerr << "\tQuantization Table Destination Selector: "
                          << (uint16_t)fcsp.QuantizationTableDestSelector
                          << std::endl;
                std::cerr
                    << "\tDC Entropy Coding Table Destination Selector: "
                    << (uint16_t)pScsp[i].DcEntropyCodingTableDestSelector()
                    << std::endl;
                std::cerr
                    << "\tAC Entropy Coding Table Destination Selector: "
                    << (uint16_t)pScsp[i].AcEntropyCodingTableDestSelector()
                    << std::endl;
#endif

                // Decode DC
                uint8_t dc_code =
                    m_treeHuffman[pScsp[i].DcEntropyCodingTableDestSelector()]
                        .DecodeSingleValue(scan_data.data(), scan_data.size(),
                                           &byte_offset, &bit_offset);
                uint8_t dc_bit_length = dc_code & 0x0F;
                int16_t dc_value;
                uint32_t tmp_value;

                if (!dc_code) {
#if DUMP_DETAILS
                    std::cerr << "Found EOB when decode DC!" << std::endl;
#endif
                    dc_value = 0;
                } else {
                    if (dc_bit_length + bit_offset <= 8) {
                        tmp_value = ((scan_data[byte_offset] &
                                      ((0x01u << (8 - bit_offset)) - 1)) >>
                                     (8 - dc_bit_length - bit_offset));
                    } else {
                        uint8_t bits_in_first_byte = 8 - bit_offset;
                        uint8_t append_full_bytes =
                            (dc_bit_length - bits_in_first_byte) / 8;
                        uint8_t bits_in_last_byte = dc_bit_length -
                                                    bits_in_first_byte -
                                                    8 * append_full_bytes;
                        tmp_value = (scan_data[byte_offset] &
                                     ((0x01u << (8 - bit_offset)) - 1));
                        for (int m = 1; m <= append_full_bytes; m++) {
                            tmp_value <<= 8;
                            tmp_value += scan_data[byte_offset + m];
                        }
                        tmp_value <<= bits_in_last_byte;
                        tmp_value +=
                            (scan_data[byte_offset + append_full_bytes + 1] >>
                             (8 - bits_in_last_byte));
                    }

                    // decode dc value
                    if ((tmp_value >> (dc_bit_length - 1)) == 0) {
                        // MSB = 1, turn it to minus value
                        dc_value = -(int16_t)(~tmp_value &
                                              ((0x0001u << dc_bit_length) - 1));
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
                while (bit_offset >= 8) {
                    bit_offset -= 8;
                    byte_offset++;
                }

                // Decode AC
                int ac_index = 1;
                while (byte_offset < scan_data.size() && ac_index < 64) {
                    uint8_t ac_code =
                        m_treeHuffman
                            [2 + pScsp[i].AcEntropyCodingTableDestSelector()]
                                .DecodeSingleValue(scan_data.data(),
                                                   scan_data.size(),
                                                   &byte_offset, &bit_offset);

                    if (!ac_code) {
#if DUMP_DETAILS
                        std::cerr << "Found EOB when decode AC!" << std::endl;
#endif
                        break;
                    }

                    if (ac_code == 0xF0) {
#if DUMP_DETAILS
                        std::cerr << "Found ZRL when decode AC!" << std::endl;
#endif
                        ac_index += 16;
                        continue;
                    }

                    uint8_t ac_zero_length = ac_code >> 4;
                    ac_index += ac_zero_length;
                    uint8_t ac_bit_length = ac_code & 0x0F;
                    int16_t ac_value;

                    if (ac_bit_length + bit_offset <= 8) {
                        tmp_value = ((scan_data[byte_offset] &
                                      ((0x01u << (8 - bit_offset)) - 1)) >>
                                     (8 - ac_bit_length - bit_offset));
                    } else {
                        uint8_t bits_in_first_byte = 8 - bit_offset;
                        uint8_t append_full_bytes =
                            (ac_bit_length - bits_in_first_byte) / 8;
                        uint8_t bits_in_last_byte = ac_bit_length -
                                                    bits_in_first_byte -
                                                    8 * append_full_bytes;
                        tmp_value = (scan_data[byte_offset] &
                                     ((0x01u << (8 - bit_offset)) - 1));
                        for (int m = 1; m <= append_full_bytes; m++) {
                            tmp_value <<= 8;
                            tmp_value += scan_data[byte_offset + m];
                        }
                        tmp_value <<= bits_in_last_byte;
                        tmp_value +=
                            (scan_data[byte_offset + append_full_bytes + 1] >>
                             (8 - bits_in_last_byte));
                    }

                    // decode ac value
                    if ((tmp_value >> (ac_bit_length - 1)) == 0) {
                        // MSB = 1, turn it to minus value
                        ac_value = -(int16_t)(~tmp_value &
                                              ((0x0001u << ac_bit_length) - 1));
                    } else {
                        ac_value = tmp_value;
                    }

#ifdef DUMP_DETAILS
                    printf("AC Code: %x\n", ac_code);
                    printf("AC Bit Length: %d\n", ac_bit_length);
                    printf("AC Value: %d\n", ac_value);
#endif

                    int index = m_zigzagIndex[ac_index];
                    block[i][index >> 3][index & 0x07] = ac_value;

                    // forward pointers to end of AC
                    bit_offset += ac_bit_length;
                    while (bit_offset >= 8) {
                        bit_offset -= 8;
                        byte_offset++;
                    }

                    ac_index++;
                }

#ifdef DUMP_DETAILS
                printf("Extracted Component[%d] 8x8 block: ", i);
                std::cerr << block[i];
#endif
                MatrixMulByElement(
                    block[i], block[i],
                    m_tableQuantization[fcsp.QuantizationTableDestSelector]);
#ifdef DUMP_DETAILS
                std::cerr << "After Quantization: " << block[i];
#endif
                block[i][0][0] += 1024.0f;  // level shift. same as +128 to each
                                            // element after IDCT
                block[i] = IDCT8X8(block[i]);
#ifdef DUMP_DETAILS
                std::cerr << "After IDCT: " << block[i];
#endif
            }

            assert(m_nComponentsInFrame <= 4);

            YCbCrf ycbcr;
            RGBf rgb;
            int mcu_index_x = mcu_index % mcu_count_x;
            int mcu_index_y = mcu_index / mcu_count_x;
            uint8_t* pBuf;

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    for (int k = 0; k < m_nComponentsInFrame; k++) {
                        ycbcr[k] = block[k][i][j];
                    }

                    pBuf = reinterpret_cast<uint8_t*>(img.data) +
                           ((ptrdiff_t)img.pitch *
                                ((ptrdiff_t)mcu_index_y * 8 + i) +
                            ((ptrdiff_t)mcu_index_x * 8 + j) *
                                (img.bitcount >> 3));
                    rgb = ConvertYCbCr2RGB(ycbcr);
                    reinterpret_cast<R8G8B8A8Unorm*>(pBuf)->data[0] =
                        (uint8_t)rgb[0];
                    reinterpret_cast<R8G8B8A8Unorm*>(pBuf)->data[1] =
                        (uint8_t)rgb[1];
                    reinterpret_cast<R8G8B8A8Unorm*>(pBuf)->data[2] =
                        (uint8_t)rgb[2];
                    reinterpret_cast<R8G8B8A8Unorm*>(pBuf)->data[3] = 255;
                }
            }

            mcu_index++;

            if (m_nRestartInterval != 0 &&
                (mcu_index % m_nRestartInterval == 0)) {
                if (bit_offset) {
                    // finish current byte
                    bit_offset = 0;
                    byte_offset++;
                }
                assert(byte_offset == scan_data.size());
                break;
            }
        }

        return scanLength;
    }

   public:
    Image Parse(Buffer& buf) override {
        Image img;

        const uint8_t* pData = buf.GetData();
        const uint8_t* pDataEnd = buf.GetData() + buf.GetDataSize();

        const auto* pFileHeader =
            reinterpret_cast<const JFIF_FILEHEADER*>(pData);
        pData += sizeof(JFIF_FILEHEADER);
        if (pFileHeader->SOI ==
            endian_net_unsigned_int((uint16_t)0xFFD8) /* FF D8 */) {
            std::cerr << "Asset is JPEG file" << std::endl;

            while (pData < pDataEnd) {
                size_t scanLength = 0;

                const auto* pSegmentHeader =
                    reinterpret_cast<const JPEG_SEGMENT_HEADER*>(pData);
#if DUMP_DETAILS
                std::cerr << "============================" << std::endl;
#endif
                switch (endian_net_unsigned_int(pSegmentHeader->Marker)) {
                    case 0xFFC0:
                    case 0xFFC2: {
                        if (endian_net_unsigned_int(pSegmentHeader->Marker) ==
                            0xFFC0)
                            std::cerr << "Start Of Frame0 (baseline DCT)"
                                      << std::endl;
                        else
                            std::cerr << "Start Of Frame2 (progressive DCT)"
                                      << std::endl;

                        std::cerr << "----------------------------"
                                  << std::endl;

                        const auto* pFrameHeader =
                            reinterpret_cast<const FRAME_HEADER*>(pData);
                        m_nSamplePrecision = pFrameHeader->SamplePrecision;
                        m_nLines = endian_net_unsigned_int(
                            (uint16_t)pFrameHeader->NumOfLines);
                        m_nSamplesPerLine = endian_net_unsigned_int(
                            (uint16_t)pFrameHeader->NumOfSamplesPerLine);
                        m_nComponentsInFrame =
                            pFrameHeader->NumOfComponentsInFrame;
                        mcu_index = 0;
                        mcu_count_x = ((m_nSamplesPerLine + 7) >> 3);
                        mcu_count_y = ((m_nLines + 7) >> 3);
                        mcu_count = mcu_count_x * mcu_count_y;

                        std::cerr << "Sample Precision: " << m_nSamplePrecision
                                  << std::endl;
                        std::cerr << "Num of Lines: " << m_nLines << std::endl;
                        std::cerr
                            << "Num of Samples per Line: " << m_nSamplesPerLine
                            << std::endl;
                        std::cerr << "Num of Components In Frame: "
                                  << m_nComponentsInFrame << std::endl;
                        std::cerr << "Total MCU count: " << mcu_count
                                  << std::endl;

                        const uint8_t* pTmp = pData + sizeof(FRAME_HEADER);
                        const auto* pFcsp = reinterpret_cast<
                            const FRAME_COMPONENT_SPEC_PARAMS*>(pTmp);
                        for (uint8_t i = 0;
                             i < pFrameHeader->NumOfComponentsInFrame; i++) {
                            std::cerr << "\tComponent Identifier: "
                                      << (uint16_t)pFcsp->ComponentIdentifier
                                      << std::endl;
                            std::cerr
                                << "\tHorizontal Sampling Factor: "
                                << (uint16_t)pFcsp->HorizontalSamplingFactor()
                                << std::endl;
                            std::cerr
                                << "\tVertical Sampling Factor: "
                                << (uint16_t)pFcsp->VerticalSamplingFactor()
                                << std::endl;
                            std::cerr
                                << "\tQuantization Table Destination Selector: "
                                << (uint16_t)
                                       pFcsp->QuantizationTableDestSelector
                                << std::endl;
                            std::cerr << std::endl;
                            m_tableFrameComponentsSpec.push_back(*pFcsp);
                            pFcsp++;
                        }

                        img.Width = m_nSamplesPerLine;
                        img.Height = m_nLines;
                        img.bitcount = 32;
                        img.pitch = mcu_count_x * 8 * (img.bitcount >> 3);
                        img.data_size = (size_t)img.pitch * mcu_count_y * 8 *
                                        (img.bitcount >> 3);
                        img.data = new uint8_t[img.data_size];

                        pData += (ptrdiff_t)endian_net_unsigned_int(
                                     pSegmentHeader->Length) +
                                 2 /* length of marker */;
                    } break;
                    case 0xFFC4: {
                        std::cerr << "Define Huffman Table" << std::endl;
                        std::cerr << "----------------------------"
                                  << std::endl;

                        size_t segmentLength =
                            endian_net_unsigned_int(pSegmentHeader->Length) - 2;

                        const uint8_t* pTmp =
                            pData + sizeof(JPEG_SEGMENT_HEADER);

                        while (segmentLength > 0) {
                            const auto* pHtable =
                                reinterpret_cast<const HUFFMAN_TABLE_SPEC*>(
                                    pTmp);
                            std::cerr
                                << "Table Class: " << pHtable->TableClass()
                                << std::endl;
                            std::cerr << "Destination Identifier: "
                                      << pHtable->DestinationIdentifier()
                                      << std::endl;

                            const uint8_t* pCodeValueStart =
                                reinterpret_cast<const uint8_t*>(pHtable) +
                                sizeof(HUFFMAN_TABLE_SPEC);

                            auto num_symbo =
                                m_treeHuffman[(pHtable->TableClass() << 1) |
                                              pHtable->DestinationIdentifier()]
                                    .PopulateWithHuffmanTable(
                                        pHtable->NumOfHuffmanCodes,
                                        pCodeValueStart);

#ifdef DUMP_DETAILS
                            m_treeHuffman[(pHtable->TableClass() << 1) |
                                          pHtable->DestinationIdentifier()]
                                .Dump();
#endif

                            size_t processed_length =
                                sizeof(HUFFMAN_TABLE_SPEC) + num_symbo;
                            pTmp += processed_length;
                            segmentLength -= processed_length;
                        }
                        pData += (ptrdiff_t)endian_net_unsigned_int(
                                     pSegmentHeader->Length) +
                                 2 /* length of marker */;
                    } break;
                    case 0xFFDB: {
                        std::cerr << "Define Quantization Table" << std::endl;
                        std::cerr << "----------------------------"
                                  << std::endl;

                        size_t segmentLength =
                            endian_net_unsigned_int(pSegmentHeader->Length) - 2;

                        const uint8_t* pTmp =
                            pData + sizeof(JPEG_SEGMENT_HEADER);

                        while (segmentLength > 0) {
                            const auto* pQtable = reinterpret_cast<
                                const QUANTIZATION_TABLE_SPEC*>(pTmp);
                            std::cerr << "Element Precision: "
                                      << pQtable->ElementPrecision()
                                      << std::endl;
                            std::cerr << "Destination Identifier: "
                                      << pQtable->DestinationIdentifier()
                                      << std::endl;

                            const uint8_t* pElementDataStart =
                                reinterpret_cast<const uint8_t*>(pQtable) +
                                sizeof(QUANTIZATION_TABLE_SPEC);

                            for (int i = 0; i < 64; i++) {
                                int index = m_zigzagIndex[i];
                                if (pQtable->ElementPrecision() == 0) {
                                    m_tableQuantization
                                        [pQtable->DestinationIdentifier()]
                                        [index >> 3][index & 0x7] =
                                            pElementDataStart[i];
                                } else {
                                    m_tableQuantization
                                        [pQtable->DestinationIdentifier()]
                                        [index >> 3][index & 0x7] =
                                            endian_net_unsigned_int(
                                                *((uint16_t*)pElementDataStart +
                                                  i));
                                }
                            }
#ifdef DUMP_DETAILS
                            std::cerr << m_tableQuantization
                                    [pQtable->DestinationIdentifier()];
#endif

                            size_t processed_length =
                                sizeof(QUANTIZATION_TABLE_SPEC) +
                                (size_t)64 *
                                    ((size_t)pQtable->ElementPrecision() + 1);
                            pTmp += processed_length;
                            segmentLength -= processed_length;
                        }
                        pData += (ptrdiff_t)endian_net_unsigned_int(
                                     pSegmentHeader->Length) +
                                 2 /* length of marker */;
                    } break;
                    case 0xFFDD: {
                        std::cerr << "Define Restart Interval" << std::endl;
                        std::cerr << "----------------------------"
                                  << std::endl;

                        auto* pRestartHeader = (RESTART_INTERVAL_DEF*)pData;
                        m_nRestartInterval = endian_net_unsigned_int(
                            (uint16_t)pRestartHeader->RestartInterval);
                        std::cerr << "Restart interval: " << m_nRestartInterval
                                  << std::endl;
                        pData += (ptrdiff_t)endian_net_unsigned_int(
                                     pSegmentHeader->Length) +
                                 2 /* length of marker */;
                    } break;
                    case 0xFFDA: {
                        std::cerr << "Start Of Scan" << std::endl;
                        std::cerr << "----------------------------"
                                  << std::endl;

                        auto* pScanHeader = (SCAN_HEADER*)pData;
                        std::cerr << "Image Conponents in Scan: "
                                  << (uint16_t)pScanHeader->NumOfComponents
                                  << std::endl;
                        assert(pScanHeader->NumOfComponents ==
                               m_nComponentsInFrame);

                        const uint8_t* pTmp = pData + sizeof(SCAN_HEADER);
                        pScsp =
                            reinterpret_cast<const SCAN_COMPONENT_SPEC_PARAMS*>(
                                pTmp);

                        const uint8_t* pScanData =
                            pData +
                            endian_net_unsigned_int(
                                (uint16_t)pScanHeader->Length) +
                            2;

                        scanLength = parseScanData(pScanData, pDataEnd, img);
                        pData += (ptrdiff_t)endian_net_unsigned_int(
                                     pSegmentHeader->Length) +
                                 2 + scanLength /* length of marker */;
                    } break;
                    case 0xFFD0:
                    case 0xFFD1:
                    case 0xFFD2:
                    case 0xFFD3:
                    case 0xFFD4:
                    case 0xFFD5:
                    case 0xFFD6:
                    case 0xFFD7: {
#if DUMP_DETAILS
                        std::cerr << "Restart Of Scan" << std::endl;
                        std::cerr << "----------------------------"
                                  << std::endl;
#endif

                        const uint8_t* pScanData = pData + 2;
                        scanLength = parseScanData(pScanData, pDataEnd, img);
                        pData += 2 + scanLength /* length of marker */;
                    } break;
                    case 0xFFD9: {
                        std::cerr << "End Of Scan" << std::endl;
                        std::cerr << "----------------------------"
                                  << std::endl;
                        pData += 2 /* length of marker */;
                    } break;
                    case 0xFFE0: {
                        const APP0* pApp0 =
                            reinterpret_cast<const APP0*>(pData);
                        switch (endian_net_unsigned_int(
                            *(uint32_t*)pApp0->Identifier)) {
                            case "JFIF\0"_u32: {
                                const auto* pJfifApp0 =
                                    reinterpret_cast<const JFIF_APP0*>(pApp0);
                                std::cerr << "JFIF-APP0" << std::endl;
                                std::cerr << "----------------------------"
                                          << std::endl;
                                std::cerr << "JFIF Version: "
                                          << (uint16_t)pJfifApp0->MajorVersion
                                          << "."
                                          << (uint16_t)pJfifApp0->MinorVersion
                                          << std::endl;
                                std::cerr
                                    << "Density Units: "
                                    << ((pJfifApp0->DensityUnits == 0)
                                            ? "No units"
                                            : ((pJfifApp0->DensityUnits == 1)
                                                   ? "Pixels per inch"
                                                   : "Pixels per centimeter"))
                                    << std::endl;
                                std::cerr << "Density: "
                                          << endian_net_unsigned_int(
                                                 pJfifApp0->Xdensity)
                                          << "*"
                                          << endian_net_unsigned_int(
                                                 pJfifApp0->Ydensity)
                                          << std::endl;
                                if (pJfifApp0->Xthumbnail &&
                                    pJfifApp0->Ythumbnail) {
                                    std::cerr << "Thumbnail Dimesions [w*h]: "
                                              << (uint16_t)pJfifApp0->Xthumbnail
                                              << "*"
                                              << (uint16_t)pJfifApp0->Ythumbnail
                                              << std::endl;
                                } else {
                                    std::cerr << "No thumbnail defined in "
                                                 "JFIF-APP0 segment!"
                                              << std::endl;
                                }
                            } break;
                            case "JFXX\0"_u32: {
                                const auto* pJfxxApp0 =
                                    reinterpret_cast<const JFXX_APP0*>(pApp0);
                                std::cerr << "Thumbnail Format: ";
                                switch (pJfxxApp0->ThumbnailFormat) {
                                    case 0x10:
                                        std::cerr << "JPEG format";
                                        break;
                                    case 0x11:
                                        std::cerr << "1 byte per pixel "
                                                     "palettized format";
                                        break;
                                    case 0x13:
                                        std::cerr
                                            << "3 byte per pixel RGB format";
                                        break;
                                    default:
                                        std::printf(
                                            "Unrecognized Thumbnail Format: "
                                            "%x\n",
                                            pJfxxApp0->ThumbnailFormat);
                                }
                                std::cerr << std::endl;
                            }
                            default:
                                std::cerr << "Ignor Unrecognized APP0 segment."
                                          << std::endl;
                        }
                        pData += (ptrdiff_t)endian_net_unsigned_int(
                                     pSegmentHeader->Length) +
                                 2 /* length of marker */;
                    } break;
                    case 0xFFFE: {
                        std::cerr << "Text Comment" << std::endl;
                        std::cerr << "----------------------------"
                                  << std::endl;
                        pData += (ptrdiff_t)endian_net_unsigned_int(
                                     pSegmentHeader->Length) +
                                 2 /* length of marker */;
                    } break;
                    default: {
                        std::printf(
                            "Ignor Unrecognized Segment. Marker=%0x\n",
                            endian_net_unsigned_int(pSegmentHeader->Marker));
                        pData += (ptrdiff_t)endian_net_unsigned_int(
                                     pSegmentHeader->Length) +
                                 2 /* length of marker */;
                    } break;
                }
            }
        } else {
            std::cerr << "File is not a JPEG file!" << std::endl;
        }

        img.mipmaps.emplace_back(img.Width, img.Height, img.pitch, 0,
                                 img.data_size);

        return img;
    }
};
}  // namespace My
