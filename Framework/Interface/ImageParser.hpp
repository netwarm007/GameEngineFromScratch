#pragma once
#include "Interface.hpp"
#include "Image.hpp"
#include "Buffer.hpp"

namespace My {
    interface ImageParser
    {
    public:
        virtual Image& Parse(const Buffer& buf) = 0;
    };
}

