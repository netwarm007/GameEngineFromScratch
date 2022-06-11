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
    switch(format) {
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
        case COMPRESSED_FORMAT::BC4S:
            out << "BC4S\n";
            break;
        case COMPRESSED_FORMAT::BC4U:
            out << "BC4U\n";
            break;
        case COMPRESSED_FORMAT::BC5S:
            out << "BC5S\n";
            break;
        case COMPRESSED_FORMAT::BC5U:
            out << "BC5U\n";
            break;
        case COMPRESSED_FORMAT::BC6S:
            out << "BC6S\n";
            break;
        case COMPRESSED_FORMAT::BC6U:
            out << "BC6U\n";
            break;
        case COMPRESSED_FORMAT::BC7S:
            out << "BC7S\n";
            break;
        case COMPRESSED_FORMAT::BC7U:
            out << "BC7U\n";
            break;
        case COMPRESSED_FORMAT::PVRTC:
            out << "PVRTC\n";
            break;
        case COMPRESSED_FORMAT::ETC:
            out << "ETC\n";
            break;
        case COMPRESSED_FORMAT::ASTC:
            out << "ASTC\n";
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

}  // namespace My
