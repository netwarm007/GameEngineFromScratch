#include "Plane.hpp"

using namespace My;
using namespace std;

void Plane::GetAabb(const Matrix4X4f& trans, Vector3f& aabbMin,
                    Vector3f& aabbMax) const {
    (void)trans;
    float minf = numeric_limits<float>::lowest();
    float maxf = numeric_limits<float>::max();
    aabbMin = {minf, minf, minf};
    aabbMax = {maxf, maxf, maxf};
}