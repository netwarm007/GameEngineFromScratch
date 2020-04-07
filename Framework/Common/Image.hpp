#pragma once
#include <cstring>
#include <iostream>
#include "config.h"
#include "geommath.hpp"

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

        Image() 
            
        {
            std::memset(mipmaps, 0x00, sizeof(mipmaps));
        };
    };

    std::ostream& operator<<(std::ostream& out, const Image& image);
}


