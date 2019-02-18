#include <cstdint>
#include <cstring>

using namespace std;

namespace Dummy
{
    void BuildIdentityMatrix(float * data, const int32_t n)
    {
        memset(data, 0x00, sizeof(float) * n);

        for(int32_t i = 0; i < n; i++)
        {
            *(data + i*n + i) = 1.0f;
        }
    }

    void MatrixExchangeYandZ(float * data, const int32_t rows, const int32_t cols)
    {
        for(int32_t row_index = 0; row_index < rows; row_index++)
        {
            uint32_t *p, *q;
            p = reinterpret_cast<uint32_t*>(data + row_index*cols + 1);
            q = reinterpret_cast<uint32_t*>(data + row_index*cols + 2);
            *p ^= *q;
            *q ^= *p;
            *p ^= *q;
        }
    }
}