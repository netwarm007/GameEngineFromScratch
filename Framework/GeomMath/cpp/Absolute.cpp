#include <cstddef>
#include <cmath>

namespace Dummy
{
    void Absolute(float * result, const float * a, const size_t count)
    {
        for (size_t i = 0; i < count; i++)
        {
            result[i] = fabs(a[i]);
        }
    }
}
