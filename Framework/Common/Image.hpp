#pragma once
#include <iostream>
#include "config.h"
#include "geommath.hpp"
#include "MemoryManager.hpp"

namespace My {

    struct Image {
        uint32_t Width;
        uint32_t Height;
        void* data;
        uint32_t bitcount;
        uint32_t pitch;
        size_t  data_size;

        Image() : Width(0),
            Height(0),
            data(nullptr),
            bitcount(0),
            pitch(0),
            data_size(0)
        {};

        ~Image() {
            if (data) g_pMemoryManager->Free(data, data_size);
        }
    };

    std::ostream& operator<<(std::ostream& out, const Image& image);
}


