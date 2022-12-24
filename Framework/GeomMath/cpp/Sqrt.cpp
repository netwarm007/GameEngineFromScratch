#include <cmath>
#include <cstddef>

using namespace std;

namespace Dummy {
void Sqrt(const float* v, const size_t count, float* result) {
    for (size_t i = 0; i < count; i++) {
        result[i] = sqrt(v[i]);
    }
}
}  // namespace Dummy