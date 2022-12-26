#include "Image.hpp"

int main() {
    My::Image img;
    img.Width = 300;
    img.Height = 300;
    img.bitcount = 24;
    img.bitdepth = 8;
    img.pixel_format = My::PIXEL_FORMAT::RGB8;
    img.pitch = (img.bitcount >> 3) * img.Width;
    img.compressed = false;
    img.compress_format = My::COMPRESSED_FORMAT::NONE;
    img.data_size = img.Width * img.Height * (img.bitcount >> 3);
    img.data = new uint8_t[img.data_size];
    std::memset(img.data, 0x00, img.data_size);

    for (int y = 0; y < img.Height; y++) {
        for (int x = 0; x < img.Width; x++) {
            switch (x % 3) {
                case 0:
                    img.SetR(x, y, 255);
                    break;
                case 1:
                    img.SetG(x, y, 255);
                    break;
                case 2:
                    img.SetB(x, y, 255);
                    break;
                default: assert(0);
            }
        }
    }

    img.SaveTGA("TgaEncoderTest.tga");

    return 0;
}