#include "ASTC.hpp"

using namespace My;

std::map<uint32_t, COMPRESSED_FORMAT> AstcParser::fmt_lut = {
    {0x040401, COMPRESSED_FORMAT::ASTC_4x4},
    {0x060601, COMPRESSED_FORMAT::ASTC_6x6},
    {0x080801, COMPRESSED_FORMAT::ASTC_6x6}
};