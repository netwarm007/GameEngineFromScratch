#pragma once
#include "config.h"
#include "geommath.hpp"
#include <cstring>
#include <iostream>

namespace My {
    struct Image {
        uint32_t Width{0};
        uint32_t Height{0};
        uint8_t* data{nullptr};
        uint32_t bitcount{0};
        uint32_t pitch{0};
        size_t  data_size{0};
        bool    compressed{false};
        bool    is_float{false};
        uint32_t compress_format{0};
        uint32_t mipmap_count{1};
        struct Mipmap {
            uint32_t Width{0};
            uint32_t Height{0};
            uint32_t pitch{0};
            size_t offset{0};
            size_t data_size{0};
        } mipmaps[10];

        Image() = default;
        Image(const Image& rhs)  = delete; // disable copy contruct
        Image(Image&& rhs) noexcept;
        Image& operator =(const Image& rhs) = delete; // disable copy assignment
        Image& operator =(Image&& rhs) noexcept;
        ~Image() { if(data) delete[] data; }
    };

    std::ostream& operator<<(std::ostream& out, const Image& image);
}


