#include <cstddef>

namespace Dummy
{
    void Normalize(const size_t count, float * v, float length)
    {
        if (!length) return;
        const double one_over_length = 1.0 / length;
        for(size_t index = 0; index < count; index++)
        {
            v[index] *= one_over_length;
        }
    }
}
