#pragma once
#include "AudioClip.hpp"
#include "Buffer.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ AudioClipParser {
   public:
    virtual ~AudioClipParser() = default;
    virtual AudioClip Parse(Buffer & buf) = 0;
};
}  // namespace My
