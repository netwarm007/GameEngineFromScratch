#pragma once
#include <cstring>
#include <iostream>
#include "config.h"
#include "geommath.hpp"

namespace My {

    struct Image {
        uint32_t Width;
        uint32_t Height;
        uint8_t* data;
        uint32_t bitcount;
        uint32_t pitch;
        size_t  data_size;
        bool    compressed;
        bool    is_float;
        uint32_t compress_format;
        uint32_t mipmap_count;
        struct Mipmap {
            uint32_t Width;
            uint32_t Height;
            uint32_t pitch;
            size_t offset;
            size_t data_size;
        } mipmaps[10];

        Image() : Width(0),
            Height(0),
            data(nullptr),
            bitcount(0),
            pitch(0),
            data_size(0),
            compressed(false),
            is_float(false),
            mipmap_count(1)
        {
            std::memset(mipmaps, 0x00, sizeof(mipmaps));
        };
    };

    std::ostream& operator<<(std::ostream& out, const Image& image);
}


