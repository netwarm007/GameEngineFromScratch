#include <iostream>

#include "Encoder/PPM.hpp"

using namespace My;

int main () {
    // Image

    Image img;
    img.Width   = 512;
    img.Height  = 512;
    img.bitcount = 24;
    img.bitdepth = 8;
    img.pixel_format = PIXEL_FORMAT::RGB8;
    img.pitch = (img.bitcount >> 3) * img.Width;
    img.compressed = false;
    img.compress_format = COMPRESSED_FORMAT::NONE;
    img.data_size = img.Width * img.Height * (img.bitcount >> 3);
    img.data = new uint8_t[img.data_size];

    // Render

    for (auto j = 0; j < img.Height; j++) {
        for (auto i = 0; i < img.Width ; i++) {
            auto r = (double)i / (img.Width - 1);
            auto g = (double)j / (img.Height - 1);
            auto b = 0.25;

            img.SetR(i, j, static_cast<int>(255.999 * r));
            img.SetG(i, j, static_cast<int>(255.999 * g));
            img.SetB(i, j, static_cast<int>(255.999 * b));
        }
    }

    PpmEncoder encoder;
    encoder.Encode(img);

    return 0;
}
