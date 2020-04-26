#pragma once
#include "Buffer.hpp"
#include "Image.hpp"
#include "Interface.hpp"

namespace My {
    Interface ImageParser
    {
    public:
        virtual ~ImageParser() = default;
        virtual Image Parse(Buffer& buf) = 0;
    };
}

