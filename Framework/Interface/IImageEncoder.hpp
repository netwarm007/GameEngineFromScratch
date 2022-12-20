#pragma once
#include "Buffer.hpp"
#include "Image.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ ImageEncoder {
   public:
    virtual ~ImageEncoder() = default;
    virtual Buffer Encode(Image & img) = 0;
};
}  // namespace My
