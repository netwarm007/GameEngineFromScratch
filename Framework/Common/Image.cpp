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
    mipmaps = std::move(rhs.mipmaps);
    rhs.Width = 0;
    rhs.Height = 0;
    rhs.data = nullptr;
    rhs.bitcount = 0;
    rhs.pitch = 0;
    rhs.data_size = 0;
    rhs.compressed = false;
    rhs.is_float = false;
    rhs.compress_format = 0;
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
        mipmaps = std::move(rhs.mipmaps);
        rhs.Width = 0;
        rhs.Height = 0;
        rhs.data = nullptr;
        rhs.bitcount = 0;
        rhs.pitch = 0;
        rhs.data_size = 0;
        rhs.compressed = false;
        rhs.is_float = false;
        rhs.compress_format = 0;
    }
    return *this;
}

ostream& operator<<(ostream& out, const Image& image) {
    out << "Image" << endl;
    out << "-----" << endl;
    out << "Width: " << image.Width << endl;
    out << "Height: " << image.Height << endl;
    out << "Bit Count: " << image.bitcount << endl;
    out << "Pitch: " << image.pitch << endl;
    out << "Data Size: " << image.data_size << endl;

#if DUMP_DETAILS
    int byte_count = image.bitcount >> 3;

    for (uint32_t i = 0; i < image.Height; i++) {
        for (uint32_t j = 0; j < image.Width; j++) {
            for (auto k = 0; k < byte_count; k++) {
                printf("%x ",
                       reinterpret_cast<uint8_t*>(
                           image.data)[image.pitch * i + j * byte_count + k]);
            }
            cout << "\t";
        }
        cout << endl;
    }
#endif

    return out;
}
}  // namespace My
