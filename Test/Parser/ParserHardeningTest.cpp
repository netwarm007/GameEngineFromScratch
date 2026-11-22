// Regression tests for the Framework parser hardening (security audit
// 2026-11): malformed asset inputs must be rejected without out-of-bounds
// reads/writes. These cases crash or corrupt memory with the old parsers.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "ASTC.hpp"
#include "AssetLoader.hpp"
#include "DDS.hpp"
#include "HDR.hpp"
#include "JPEG.hpp"
#include "PNG.hpp"
#include "TGA.hpp"
#include "WAVE.hpp"

using namespace My;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAILED: " #cond " at line " << __LINE__ << "\n"; \
            return 1;                                                       \
        }                                                                   \
    } while (0)

namespace {

Buffer MakeBuffer(std::vector<uint8_t> bytes) {
    Buffer buffer(bytes.size());
    if (!bytes.empty()) {
        memcpy(buffer.GetData(), bytes.data(), bytes.size());
    }
    return buffer;
}

// Parse() takes a non-const Buffer&, so buffers must be materialized first.
template <typename Parser>
Image ParseImage(Parser& parser, std::vector<uint8_t> bytes) {
    Buffer buf = MakeBuffer(std::move(bytes));
    return parser.Parse(buf);
}

template <typename Parser>
AudioClip ParseAudio(Parser& parser, std::vector<uint8_t> bytes) {
    Buffer buf = MakeBuffer(std::move(bytes));
    return parser.Parse(buf);
}

template <typename Parser>
Image ParseString(Parser& parser, const std::string& s) {
    return ParseImage(parser,
                      std::vector<uint8_t>(s.begin(), s.end()));
}

// ---- JPEG -----------------------------------------------------------------
// A baseline frame header whose component count exceeds the 4 supported
// components used to index fixed-size stack arrays during scan decoding.
std::vector<uint8_t> MakeJpegWithComponentCount(uint8_t component_count) {
    std::vector<uint8_t> f;
    auto push = [&](uint8_t b) { f.push_back(b); };
    push(0xFF); push(0xD8);  // SOI
    // SOF0: precision 8, 8x8 image, N components
    size_t sof_len = 8 + (size_t)3 * component_count;
    push(0xFF); push(0xC0);
    push((uint8_t)(sof_len >> 8)); push((uint8_t)(sof_len & 0xFF));
    push(8);                        // precision
    push(0); push(8);               // 8 lines
    push(0); push(8);               // 8 samples per line
    push(component_count);          // number of components
    for (uint8_t i = 0; i < component_count; i++) {
        push(i);        // identifier
        push(0x11);     // sampling factors
        push(0);        // quant table 0
    }
    // DQT: one 8-bit table, id 0 (payload 65 bytes, segment length 67)
    push(0xFF); push(0xDB);
    push(0x00); push(0x43);
    push(0x00);
    for (int i = 0; i < 64; i++) push(16);
    // DHT: DC table class 0 id 0, one code of length 1, symbol 0 (EOB)
    push(0xFF); push(0xC4);
    push(0x00); push(0x11);
    push(0x00);
    push(1);
    for (int i = 1; i < 16; i++) push(0);
    push(0x00);
    // DHT: AC table class 1 id 0, one code of length 1, symbol 0 (EOB)
    push(0xFF); push(0xC4);
    push(0x00); push(0x11);
    push(0x10);
    push(1);
    for (int i = 1; i < 16; i++) push(0);
    push(0x00);
    // SOS: 1 component, DC/AC selectors 0/0, full spectral band
    push(0xFF); push(0xDA);
    push(0x00); push(0x0C);
    push(1);        // one component in scan
    push(0);        // component selector
    push(0x00);     // DC/AC table selectors
    push(0); push(63); push(0);
    // entropy data: 0x00 decodes EOB for the single MCU
    push(0x00);
    push(0xFF); push(0xD9);  // EOI
    return f;
}

// ---- PNG ------------------------------------------------------------------
void PushPngChunk(std::vector<uint8_t>& f, const char type[4],
                  const std::vector<uint8_t>& payload) {
    uint32_t len = (uint32_t)payload.size();
    f.insert(f.end(), {(uint8_t)(len >> 24), (uint8_t)(len >> 16),
                       (uint8_t)(len >> 8), (uint8_t)len});
    f.insert(f.end(), type, type + 4);
    f.insert(f.end(), payload.begin(), payload.end());
    // CRC is not validated by the parser; append a stub
    f.insert(f.end(), {0, 0, 0, 0});
}

std::vector<uint8_t> MakePngIhdr(uint32_t width, uint32_t height,
                                 uint8_t bit_depth, uint8_t color_type) {
    std::vector<uint8_t> p;
    auto put = [&](uint32_t v, int n) {
        for (int i = n - 1; i >= 0; i--) p.push_back((uint8_t)(v >> (8 * i)));
    };
    put(width, 4);
    put(height, 4);
    put(bit_depth, 1);
    put(color_type, 1);
    put(0, 1);  // compression
    put(0, 1);  // filter
    put(0, 1);  // interlace
    return p;
}

// ---- TGA ------------------------------------------------------------------
std::vector<uint8_t> MakeTgaHeader(uint16_t width, uint16_t height,
                                   uint8_t pixel_depth, uint8_t alpha_depth,
                                   uint8_t id_length = 0) {
    // TGA_FILEHEADER: IDLength(0) ColorMapType(1) ImageType(2)
    // ColorMapSpec[5](3..7) ImageSpec[10](8..17); ImageSpec holds origin at
    // 0..3, width at 4..5, height at 6..7, depth at 8, descriptor at 9.
    std::vector<uint8_t> f(18, 0);
    f[0] = id_length;
    f[2] = 2;  // uncompressed true-color
    f[12] = (uint8_t)(width & 0xFF);
    f[13] = (uint8_t)(width >> 8);
    f[14] = (uint8_t)(height & 0xFF);
    f[15] = (uint8_t)(height >> 8);
    f[16] = pixel_depth;
    f[17] = alpha_depth;
    return f;
}

// ---- DDS ------------------------------------------------------------------
std::vector<uint8_t> MakeDds(uint32_t width, uint32_t height,
                             uint32_t mip_count, const char fourcc[4],
                             size_t payload_size) {
    std::vector<uint8_t> f;
    auto put = [&](uint32_t v) {
        f.insert(f.end(),
                 {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16),
                  (uint8_t)(v >> 24)});
    };
    auto put_tag = [&](const char tag[4]) {
        f.insert(f.end(), tag, tag + 4);
    };
    put_tag("DDS ");  // magic
    put(124);         // dwSize
    put(0x1007);      // flags
    put(height);
    put(width);
    put(0);        // pitch or linear size
    put(0);        // depth
    put(mip_count);
    for (int i = 0; i < 11; i++) put(0);  // reserved1
    put(32);            // ddspf.dwSize
    put(0x4);           // DDPF_FOURCC
    put_tag(fourcc);    // dwFourCC
    put(0);             // dwRGBBitCount
    put(0); put(0); put(0); put(0);  // R/G/B/A masks
    put(0x1000); put(0); put(0); put(0);  // caps
    put(0);                               // reserved2
    f.resize(f.size() + payload_size, 0x5A);
    return f;
}

// ---- ASTC -----------------------------------------------------------------
std::vector<uint8_t> MakeAstc(uint32_t w, uint32_t h, uint8_t bx, uint8_t by,
                              size_t extra_payload) {
    // astc_header: magic[4], block xyz, dim_x[3], dim_y[3], dim_z[3] = 16 B
    std::vector<uint8_t> f(16, 0);
    f[0] = 0x13; f[1] = 0xAB; f[2] = 0xA1; f[3] = 0x5C;
    f[4] = bx; f[5] = by; f[6] = 1;
    f[7] = (uint8_t)w; f[8] = (uint8_t)(w >> 8); f[9] = (uint8_t)(w >> 16);
    f[10] = (uint8_t)h; f[11] = (uint8_t)(h >> 8); f[12] = (uint8_t)(h >> 16);
    f[13] = 1;  // dim_z[0]
    for (size_t i = 0; i < extra_payload; i++) f.push_back(0);
    return f;
}

// ---- WAVE -----------------------------------------------------------------
std::vector<uint8_t> MakeWav(uint32_t data_chunk_size, size_t actual_bytes) {
    std::vector<uint8_t> f;
    auto put32 = [&](uint32_t v) {
        f.insert(f.end(),
                 {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16),
                  (uint8_t)(v >> 24)});
    };
    auto put16 = [&](uint16_t v) {
        f.insert(f.end(), {(uint8_t)v, (uint8_t)(v >> 8)});
    };
    f.insert(f.end(), {'R', 'I', 'F', 'F'});
    put32(4 + 8 + 16 + 8 + (uint32_t)actual_bytes);
    f.insert(f.end(), {'W', 'A', 'V', 'E'});
    // fmt chunk (16 bytes payload)
    f.insert(f.end(), {'f', 'm', 't', ' '});
    put32(16);
    put16(1); put16(1); put32(44100); put32(44100); put16(1); put16(8);
    // data chunk with a declared size that may exceed the file
    f.insert(f.end(), {'d', 'a', 't', 'a'});
    put32(data_chunk_size);
    for (size_t i = 0; i < actual_bytes; i++) f.push_back(0x7F);
    return f;
}

// ---- HDR ------------------------------------------------------------------
std::string MakeHdr(size_t width, size_t height, size_t payload_rows,
                    bool y_first = true) {
    std::string s = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n";
    if (y_first) {
        s += "-Y " + std::to_string(height) + " +X " + std::to_string(width) +
             "\n";
    } else {
        s += "+X " + std::to_string(width) + " -Y " +
             std::to_string(height) + "\n";
    }
    // flat scanlines: rgbe triple with zero exponent -> 0.0f
    for (size_t r = 0; r < payload_rows; r++) {
        for (size_t x = 0; x < width; x++) {
            s.push_back((char)1);
            s.push_back((char)1);
            s.push_back((char)1);
            s.push_back((char)0);
        }
    }
    return s;
}

Buffer MakeRawBuffer(const std::string& s) {
    return MakeBuffer(std::vector<uint8_t>(s.begin(), s.end()));
}

}  // namespace

int main() {
    // ---- JPEG ---------------------------------------------------------------
    {
        JfifParser parser;
        Image img = ParseImage(parser, MakeJpegWithComponentCount(1));
        CHECK(img.data != nullptr);
        CHECK(img.Width == 8 && img.Height == 8);

        // > 4 components must be rejected outright (stack overflow before)
        CHECK(ParseImage(parser, MakeJpegWithComponentCount(5)).data ==
              nullptr);

        Buffer empty;
        CHECK(parser.Parse(empty).data == nullptr);

        // truncated SOF: declared component records exceed the segment
        std::vector<uint8_t> trunc_sof = {0xFF, 0xD8, 0xFF, 0xC0,
                                          0x00, 0x0A, 0x08, 0x00,
                                          0x08, 0x00, 0x08, 0x03};
        CHECK(ParseImage(parser, trunc_sof).data == nullptr);
    }

    // ---- PNG ----------------------------------------------------------------
    {
        PngParser parser;
        // IHDR height 0x10001: the old parser truncated it to uint16_t = 1
        // and then defiltered the whole (huge) decompressed payload into a
        // 1-row buffer. Reject instead.
        std::vector<uint8_t> f = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A,
                                  0x0A};
        PushPngChunk(f, "IHDR", MakePngIhdr(1, 0x10001, 8, 6));
        PushPngChunk(f, "IDAT", {0x78, 0x9C, 0x62, 0x00, 0x00, 0x00, 0x05,
                                 0x00, 0x01});
        PushPngChunk(f, "IEND", {});
        CHECK(ParseImage(parser, f).data == nullptr);

        // IDAT claims more bytes than the file holds
        std::vector<uint8_t> h = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A,
                                  0x0A};
        PushPngChunk(h, "IHDR", MakePngIhdr(1, 1, 8, 6));
        h.insert(h.end(),
                 {0x00, 0x00, 0xFF, 0xFF, 'I', 'D', 'A', 'T'});  // huge len
        h.insert(h.end(), {1, 2, 3, 4, 5, 6, 7, 8});
        CHECK(ParseImage(parser, h).data == nullptr);

        Buffer empty;
        CHECK(parser.Parse(empty).data == nullptr);
    }

    // ---- TGA ----------------------------------------------------------------
    {
        TgaParser parser;
        // 65535x65535 in a tiny file must be rejected (multi-GB OOB read
        // before)
        auto f = MakeTgaHeader(65535, 65535, 24, 0);
        f.resize(64, 0);
        CHECK(ParseImage(parser, f).data == nullptr);

        // unsupported pixel depth rejected
        auto bad_depth = MakeTgaHeader(2, 2, 13, 0);
        for (int i = 0; i < 12; i++) bad_depth.push_back(0);
        CHECK(ParseImage(parser, bad_depth).data == nullptr);

        // image ID longer than the file
        auto bad_id = MakeTgaHeader(2, 2, 24, 0, 200);
        CHECK(ParseImage(parser, bad_id).data == nullptr);

        // well-formed 2x2 RGB24 accepted
        auto ok = MakeTgaHeader(2, 2, 24, 0);
        for (int i = 0; i < 12; i++) ok.push_back((uint8_t)i);
        Image img = ParseImage(parser, ok);
        CHECK(img.data != nullptr && img.Width == 2 && img.Height == 2);

        // 32bpp with descriptor alpha bits == 0 (RGBX, as shipped in
        // icelogo-*.tga) must still decode: the pixel layout is decided by the
        // pixel depth, not the unreliable alpha count.
        auto rgbx = MakeTgaHeader(2, 2, 32, 0);
        for (int i = 0; i < 16; i++) rgbx.push_back((uint8_t)i);
        Image img32 = ParseImage(parser, rgbx);
        CHECK(img32.data != nullptr && img32.Width == 2 && img32.Height == 2);

        Buffer empty;
        CHECK(parser.Parse(empty).data == nullptr);
    }

    // ---- DDS ----------------------------------------------------------------
    {
        DdsParser parser;
        // mip chain claims far more data than the file holds (OOB memcpy
        // before)
        auto f = MakeDds(256, 256, 32, "DXT1", 16);
        CHECK(ParseImage(parser, f).data == nullptr);

        // well-formed single-level DXT1 (4x4 block = 8 bytes)
        auto ok = MakeDds(4, 4, 1, "DXT1", 8);
        Image img = ParseImage(parser, ok);
        CHECK(img.data != nullptr && img.compressed);

        Buffer empty;
        CHECK(parser.Parse(empty).data == nullptr);
    }

    // ---- ASTC ---------------------------------------------------------------
    {
        AstcParser parser;
        // truncated header (wild read / bad_alloc before)
        std::vector<uint8_t> tiny = {0x13, 0xAB, 0xA1, 0x5C};
        CHECK(ParseImage(parser, tiny).data == nullptr);

        // block data shorter than the declared grid
        auto f = MakeAstc(1024, 1024, 4, 4, 64);
        CHECK(ParseImage(parser, f).data == nullptr);

        // well-formed single 4x4 block
        auto ok = MakeAstc(4, 4, 4, 4, 16);
        Image img = ParseImage(parser, ok);
        CHECK(img.data != nullptr && img.compressed);
    }

    // ---- WAVE ---------------------------------------------------------------
    {
        WaveParser parser;
        // data chunk claims 4 GB in a tiny file (over-read by consumers
        // before)
        CHECK(ParseAudio(parser, MakeWav(0xFFFFFFFFu, 8)).data_length == 0);

        // well-formed 8-bit mono, 4 samples
        AudioClip clip = ParseAudio(parser, MakeWav(4, 4));
        CHECK(clip.data != nullptr && clip.data_length == 4);

        Buffer empty;
        CHECK(parser.Parse(empty).data_length == 0);
    }

    // ---- HDR ----------------------------------------------------------------
    {
        HdrParser parser;
        // RLE run longer than the scanline must be rejected (heap overflow
        // before)
        std::string s =
            "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 16\n";
        uint8_t scan_hdr[4] = {2, 2, 0, 16};
        s.append((char*)scan_hdr, 4);
        s.push_back((char)200);  // run of 72 > 16
        s.push_back((char)0x7F);
        CHECK(ParseString(parser, s).data == nullptr);

        // truncated flat data (1 of 2 rows)
        CHECK(ParseString(parser, MakeHdr(16, 2, 1)).data == nullptr);

        // well-formed 16x2 flat file decodes fully
        Image img = ParseString(parser, MakeHdr(16, 2, 2));
        CHECK(img.data != nullptr && img.Width == 16 && img.Height == 2);

        // +X-first dimension line also accepted
        Image img2 = ParseString(parser, MakeHdr(16, 2, 2, false));
        CHECK(img2.data != nullptr && img2.Width == 16 && img2.Height == 2);

        // unterminated header line rejected (infinite loop before)
        CHECK(ParseString(parser, "#?RADIANCE\nFORMAT=x").data == nullptr);
    }

    // ---- AssetLoader path guard ---------------------------------------------
    {
        AssetLoader loader;
        // a real file under Asset/ resolves; traversal does not
        FILE* fp = fopen("Asset/hardening_probe.txt", "wb");
        if (fp == nullptr) {
            // running outside a directory containing Asset/ is fine too
            std::cerr << "Asset/ not present; skipping path guard checks"
                      << std::endl;
        } else {
            fputs("probe", fp);
            fclose(fp);

            CHECK(!loader.GetFileRealPath("hardening_probe.txt").empty());
            CHECK(loader.GetFileRealPath("../hardening_probe.txt").empty());
            CHECK(loader.GetFileRealPath("..\\hardening_probe.txt").empty());
            CHECK(loader.GetFileRealPath("/etc/passwd").empty());
            CHECK(loader.GetFileRealPath("C:/Windows/win.ini").empty());
            CHECK(loader.GetFileRealPath("a/../../b").empty());
            CHECK(loader.OpenFile("../hardening_probe.txt",
                                  IAssetLoader::MY_OPEN_BINARY) == nullptr);

            remove("Asset/hardening_probe.txt");
        }
    }

    std::cout << "ParserHardeningTest passed" << std::endl;
    return 0;
}
