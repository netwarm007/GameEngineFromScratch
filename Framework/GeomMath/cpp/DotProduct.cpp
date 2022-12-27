#include <cstddef>
namespace Dummy {
    void DotProduct(const float* a, const float* b, float* result,
                    const size_t count) {
        for (int i = 0; i < count; i++) {
            *result += a[i] * b[i];
        }
    }
}