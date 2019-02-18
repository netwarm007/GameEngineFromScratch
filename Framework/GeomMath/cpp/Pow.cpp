#include <cstddef>
#include <cmath>

using namespace std;

namespace Dummy
{
    void Pow(const float * v, const size_t count, const float exponent, float * result)
    {
        for (size_t i = 0; i < count; i++)
        {
            result[i] = pow(v[i], exponent);
        }
    }
}