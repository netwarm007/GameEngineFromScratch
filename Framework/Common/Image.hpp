#pragma once
#include <iostream>
#include "geommath.hpp"

namespace My {

    struct Image {
        uint32_t Width;
        uint32_t Height;
        void* data;
        uint32_t bitcount;
        uint32_t pitch;
        size_t  data_size;
    };

    std::ostream& operator<<(std::ostream& out, const Image& image);
}


