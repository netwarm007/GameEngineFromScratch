#pragma once
#include <cstring>
#include <iostream>
#include <vector>

#include "config.h"
#include "geommath.hpp"

namespace My {
struct Image {
    uint32_t Width{0};
    uint32_t Height{0};
    uint8_t* data{nullptr};
    uint32_t bitcount{0};
    size_t pitch{0};
    size_t data_size{0};
    bool compressed{false};
    bool is_float{false};
    uint32_t compress_format{0};
    struct Mipmap {
        uint32_t Width{0};
        uint32_t Height{0};
        size_t pitch{0};
        size_t offset{0};
        size_t data_size{0};

        Mipmap(uint32_t width, uint32_t height, size_t pitch_, size_t offset_,
               size_t data_size_) {
            Width = width;
            Height = height;
            pitch = pitch_;
            offset = offset_;
            data_size = data_size_;
        }
    };
    std::vector<Mipmap> mipmaps;

    Image() = default;
    Image(const Image& rhs) = delete;  // disable copy contruct
    Image(Image&& rhs) noexcept;
    Image& operator=(const Image& rhs) = delete;  // disable copy assignment
    Image& operator=(Image&& rhs) noexcept;
    ~Image() {
        if (data) delete[] data;
    }
};

std::ostream& operator<<(std::ostream& out, const Image& image);
}  // namespace My
