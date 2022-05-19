#pragma once
#include "Buffer.hpp"
#include "Image.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ ImageParser {
   public:
    virtual ~ImageParser() = default;
    virtual Image Parse(Buffer & buf) = 0;
};
}  // namespace My
