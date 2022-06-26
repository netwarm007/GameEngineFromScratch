#include "Image.hpp"

using namespace std;

namespace My {
Image::Image(Image&& rhs) noexcept {
    Width = rhs.Width;
    Height = rhs.Height;
    data = rhs.data;
    bitcount = rhs.bitcount;
    pitch = rhs.pitch;
    data_size = rhs.data_size;
    compressed = rhs.compressed;
    is_float = rhs.is_float;
    compress_format = rhs.compress_format;
    bitdepth = rhs.bitdepth;
    pixel_format = rhs.pixel_format;
    is_signed = rhs.is_signed;
    mipmaps = std::move(rhs.mipmaps);
    rhs.data = nullptr;
}

Image& Image::operator=(Image&& rhs) noexcept {
    if (this != &rhs) {
        Width = rhs.Width;
        Height = rhs.Height;
        data = rhs.data;
        bitcount = rhs.bitcount;
        pitch = rhs.pitch;
        data_size = rhs.data_size;
        compressed = rhs.compressed;
        is_float = rhs.is_float;
        compress_format = rhs.compress_format;
        bitdepth = rhs.bitdepth;
        pixel_format = rhs.pixel_format;
        is_signed = rhs.is_signed;
        mipmaps = std::move(rhs.mipmaps);
        rhs.data = nullptr;
    }
    return *this;
}

std::ostream& operator<<(std::ostream& out, COMPRESSED_FORMAT format) {
    switch (format) {
        case COMPRESSED_FORMAT::NONE:
            out << "NONE\n";
            break;
        case COMPRESSED_FORMAT::DXT1:
            out << "DXT1\n";
            break;
        case COMPRESSED_FORMAT::DXT2:
            out << "DXT2\n";
            break;
        case COMPRESSED_FORMAT::DXT3:
            out << "DXT3\n";
            break;
        case COMPRESSED_FORMAT::DXT4:
            out << "DXT4\n";
            break;
        case COMPRESSED_FORMAT::DXT5:
            out << "DXT5\n";
            break;
        case COMPRESSED_FORMAT::DXT10:
            out << "DXT10\n";
            break;
        case COMPRESSED_FORMAT::BC1:
            out << "BC1\n";
            break;
        case COMPRESSED_FORMAT::BC1A:
            out << "BC1A\n";
            break;
        case COMPRESSED_FORMAT::BC2:
            out << "BC2\n";
            break;
        case COMPRESSED_FORMAT::BC3:
            out << "BC3\n";
            break;
        case COMPRESSED_FORMAT::BC4:
            out << "BC4\n";
            break;
        case COMPRESSED_FORMAT::BC5:
            out << "BC5\n";
            break;
        case COMPRESSED_FORMAT::BC6H:
            out << "BC6H\n";
            break;
        case COMPRESSED_FORMAT::BC7:
            out << "BC7\n";
            break;
        case COMPRESSED_FORMAT::PVRTC:
            out << "PVRTC\n";
            break;
        case COMPRESSED_FORMAT::ETC:
            out << "ETC\n";
            break;
        case COMPRESSED_FORMAT::ASTC_4x4:
            out << "ASTC 4x4\n";
            break;
        case COMPRESSED_FORMAT::ASTC_5x4:
            out << "ASTC 5x4\n";
            break;
        case COMPRESSED_FORMAT::ASTC_5x5:
            out << "ASTC 5x5\n";
            break;
        case COMPRESSED_FORMAT::ASTC_6x5:
            out << "ASTC 6x5\n";
            break;
        case COMPRESSED_FORMAT::ASTC_6x6:
            out << "ASTC 6x6\n";
            break;
        case COMPRESSED_FORMAT::ASTC_8x5:
            out << "ASTC 8x5\n";
            break;
        case COMPRESSED_FORMAT::ASTC_8x6:
            out << "ASTC 8x6\n";
            break;
        case COMPRESSED_FORMAT::ASTC_8x8:
            out << "ASTC 8x8\n";
            break;
        case COMPRESSED_FORMAT::ASTC_10x5:
            out << "ASTC 10x5\n";
            break;
        case COMPRESSED_FORMAT::ASTC_10x6:
            out << "ASTC 10x6\n";
            break;
        case COMPRESSED_FORMAT::ASTC_10x8:
            out << "ASTC 10x8\n";
            break;
        case COMPRESSED_FORMAT::ASTC_10x10:
            out << "ASTC 10x10\n";
            break;
        case COMPRESSED_FORMAT::ASTC_12x10:
            out << "ASTC 12x10\n";
            break;
        case COMPRESSED_FORMAT::ASTC_12x12:
            out << "ASTC 12x12\n";
            break;
        case COMPRESSED_FORMAT::ASTC_3x3x3:
            out << "ASTC 3x3x3\n";
            break;
        case COMPRESSED_FORMAT::ASTC_4x3x3:
            out << "ASTC 4x3x3\n";
            break;
        case COMPRESSED_FORMAT::ASTC_4x4x3:
            out << "ASTC 4x4x3\n";
            break;
        case COMPRESSED_FORMAT::ASTC_4x4x4:
            out << "ASTC 4x4x4\n";
            break;
        case COMPRESSED_FORMAT::ASTC_5x4x4:
            out << "ASTC 5x4x4\n";
            break;
        case COMPRESSED_FORMAT::ASTC_5x5x4:
            out << "ASTC 5x5x4\n";
            break;
        case COMPRESSED_FORMAT::ASTC_5x5x5:
            out << "ASTC 5x5x5\n";
            break;
        case COMPRESSED_FORMAT::ASTC_6x5x5:
            out << "ASTC 6x5x5\n";
            break;
        case COMPRESSED_FORMAT::ASTC_6x6x5:
            out << "ASTC 6x6x5\n";
            break;
        case COMPRESSED_FORMAT::ASTC_6x6x6:
            out << "ASTC 6x6x6\n";
            break;
        case COMPRESSED_FORMAT::UNKNOWN:
            out << "UNKNOWN\n";
            break;
        default:
            assert(0);
    }

    return out;
}

ostream& operator<<(ostream& out, const Image& image) {
    out << "Image" << endl;
    out << "-----" << endl;
    out << "Width: " << image.Width << endl;
    out << "Height: " << image.Height << endl;
    out << "Bit Count: " << image.bitcount << endl;
    out << "Pitch: " << image.pitch << endl;
    out << "Data Size: " << image.data_size << endl;
    out << "Compressed: " << image.compressed << endl;
    out << "Compressed Format: " << image.compress_format << endl;

    return out;
}

void adjust_image(Image& image) {
    if (!image.compressed) {
        if (image.pixel_format == PIXEL_FORMAT::RGB8) {
            // DXGI does not have 24bit formats so we have to extend it to 32bit
            auto new_pitch = image.pitch / 3 * 4;
            auto data_size = (size_t)new_pitch * image.Height;
            auto* data = new uint8_t[data_size];
            uint8_t* buf;
            uint8_t* src;
            for (decltype(image.Height) row = 0; row < image.Height; row++) {
                buf = data + (ptrdiff_t)row * new_pitch;
                src = image.data + (ptrdiff_t)row * image.pitch;
                for (decltype(image.Width) col = 0; col < image.Width; col++) {
                    memcpy(buf, src, 3);
                    memset(buf + 3, 0xFF, 1);  // set alpha to 255
                    buf += 4;
                    src += 3;
                }
            }

            delete[] image.data;
            image.data = data;
            image.data_size = data_size;
            image.pitch = new_pitch;
            image.bitcount = 32;
            image.pixel_format = PIXEL_FORMAT::RGBA8;

            // adjust mipmaps
            for (auto& mip : image.mipmaps) {
                mip.pitch = mip.pitch / 3 * 4;
                mip.offset = mip.offset / 3 * 4;
                mip.data_size = mip.data_size / 3 * 4;
            }
        } else if (image.pixel_format == PIXEL_FORMAT::RGB16) {
            // DXGI does not have 48bit formats so we have to extend it to 64bit
            auto new_pitch = image.pitch / 3 * 4;
            auto data_size = new_pitch * image.Height;
            auto* data = new uint8_t[data_size];
            uint8_t* buf;
            uint8_t* src;
            for (decltype(image.Height) row = 0; row < image.Height; row++) {
                buf = data + (ptrdiff_t)row * new_pitch;
                src = image.data + (ptrdiff_t)row * image.pitch;
                for (decltype(image.Width) col = 0; col < image.Width; col++) {
                    memcpy(buf, src, 6);
                    *reinterpret_cast<int16_t*>(buf + 6) = 0b0000'0100'0000'0000; // set alpha to (fp16)1.0 = 0b 0 00001 0000 0000 00
                    buf += 8;
                    src += 6;
                }
            }

            delete[] image.data;
            image.data = data;
            image.data_size = data_size;
            image.pitch = new_pitch;
            image.bitcount = 64;
            image.pixel_format = PIXEL_FORMAT::RGBA16;

            // adjust mipmaps
            for (auto& mip : image.mipmaps) {
                mip.pitch = mip.pitch / 3 * 4;
                mip.offset = mip.offset / 3 * 4;
                mip.data_size = mip.data_size / 3 * 4;
            }
        } else if (image.pixel_format == PIXEL_FORMAT::RGB32) {
            // DXGI does not have 48bit formats so we have to extend it to 64bit
            auto new_pitch = image.pitch / 3 * 4;
            auto data_size = new_pitch * image.Height;
            auto* data = new uint8_t[data_size];
            uint8_t* buf;
            uint8_t* src;
            for (decltype(image.Height) row = 0; row < image.Height; row++) {
                buf = data + (ptrdiff_t)row * new_pitch;
                src = image.data + (ptrdiff_t)row * image.pitch;
                for (decltype(image.Width) col = 0; col < image.Width; col++) {
                    memcpy(buf, src, 12);
                    *reinterpret_cast<float*>(buf + 12) = 1.0f;
                    buf += 16;
                    src += 12;
                }
            }

            delete[] image.data;
            image.data = data;
            image.data_size = data_size;
            image.pitch = new_pitch;
            image.bitcount = 128;
            image.pixel_format = PIXEL_FORMAT::RGBA32;

            // adjust mipmaps
            for (auto& mip : image.mipmaps) {
                mip.pitch = mip.pitch / 3 * 4;
                mip.offset = mip.offset / 3 * 4;
                mip.data_size = mip.data_size / 3 * 4;
            }
        }
    }
}
}  // namespace My
