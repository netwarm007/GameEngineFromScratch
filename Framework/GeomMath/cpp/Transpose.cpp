#include <cstdint>

namespace Dummy
{
    void Transpose(const float * a, float * r, const uint32_t row_count, const uint32_t column_count)
    {
        for(uint32_t i = 0; i < row_count; i++)
        {
            for(uint32_t j = 0; j < column_count; j++)
            {
                r[j*row_count+i] = a[i*column_count+j];
            }
        }
    }
}
