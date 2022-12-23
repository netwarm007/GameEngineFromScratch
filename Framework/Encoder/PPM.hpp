#pragma once
#include "IImageEncoder.hpp"

namespace My {
    class PpmEncoder : _implements_ ImageEncoder {
        public:
        Buffer Encode(Image & img) override {
            Buffer buf;

            std::cout << "P3\n" << img.Width << ' ' << img.Height << "\n255\n";

            for (int y = img.Height - 1; y >= 0; y--) {
                for (int x = 0; x < img.Width; x++) {
                    std::cout << img.GetR(x, y) << ' ' << img.GetG(x, y) << ' ' << img.GetB(x, y) << '\n';
                }
            }

            return buf;
        }
    };
}