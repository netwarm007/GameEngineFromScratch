#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>

#include "HitableList.hpp"

using namespace My;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAILED: " #cond " at line " << __LINE__ << "\n"; \
            return 1;                                                       \
        }                                                                   \
    } while (0)

namespace {
class TestSphere : public Hitable<float> {
   public:
    TestSphere(Vector3f c, float r)
        : Hitable<float>(HitableType::kGeometry), center(c), radius(r) {}

    bool Intersect(const Ray<float>& r, Hit<float>& h, float tmin,
                   float tmax) const override {
        Vector3f oc = r.getOrigin() - center;
        float b = DotProduct(oc, r.getDirection());
        float c = DotProduct(oc, oc) - radius * radius;
        float disc = b * b - c;
        if (disc < 0) return false;
        float t = -b - std::sqrt(disc);
        if (t < tmin || t > tmax) return false;
        Vector3f p = r.pointAtParameter(t);
        h.set(t, p, (p - center) * (1.0f / radius), true, this);
        return true;
    }

    bool GetAabb(const Matrix4X4<float>&, AaBb<float, 3>& aabb) const override {
        aabb = AaBb<float, 3>(center - Vector3f({radius, radius, radius}),
                              center + Vector3f({radius, radius, radius}));
        return true;
    }

   private:
    std::ostream& dump(std::ostream& out) const override { return out; }

    Vector3f center;
    float radius;
};
}  // namespace

int main() {
    HitableList<float> list;
    AaBb<float, 3> box;
    CHECK(!list.GetAabb(BuildIdentityMatrix4X4<float>(), box));

    list.add(std::make_shared<TestSphere>(Vector3f({10, 0, 0}), 1.0f));
    list.add(std::make_shared<TestSphere>(Vector3f({4, 0, 0}), 1.0f));
    CHECK(list.size() == 2);

    // The nearer sphere (x = 4) is the one reported.
    Ray<float> ray(Vector3f({0, 0, 0}), Vector3f({1, 0, 0}));
    Hit<float> hit;
    CHECK(list.Intersect(ray, hit, 0.001f, 100.0f));
    CHECK(std::fabs(hit.getT() - 3.0f) < 1e-4f);
    CHECK(std::fabs(hit.getNormal()[0] + 1.0f) < 1e-4f);

    // Limiting tmax below the nearest hit misses both.
    CHECK(!list.Intersect(ray, hit, 0.001f, 2.0f));
    CHECK(!list.Intersect(Ray<float>(Vector3f({0, 5, 0}), Vector3f({1, 0, 0})),
                          hit, 0.001f, 100.0f));

    CHECK(list.GetAabb(BuildIdentityMatrix4X4<float>(), box));
    CHECK(box.min_point()[0] == 3.0f && box.max_point()[0] == 11.0f);
    CHECK(box.min_point()[1] == -1.0f && box.max_point()[1] == 1.0f);

    // Base-class helpers derived from the Aabb.
    Vector3f center;
    float radius = 0;
    list.GetBoundingSphere(center, radius);
    CHECK(std::fabs(center[0] - 7.0f) < 1e-4f);
    CHECK(radius > 4.0f);
    CHECK(list.GetAngularMotionDisc() > radius);

    AaBb<float, 3> temporal;
    list.CalculateTemporalAabb(BuildIdentityMatrix4X4<float>(),
                               Vector3f({2, 0, 0}), Vector3f({0, 0, 0}), 1.0f,
                               temporal);
    CHECK(temporal.max_point()[0] == 13.0f && temporal.min_point()[0] == 3.0f);
    list.CalculateTemporalAabb(BuildIdentityMatrix4X4<float>(),
                               Vector3f({-2, 0, 0}), Vector3f({0, 0, 0}), 1.0f,
                               temporal);
    CHECK(temporal.min_point()[0] == 1.0f && temporal.max_point()[0] == 11.0f);

    std::ostringstream os;
    os << list;
    CHECK(os.str().find("List") != std::string::npos);

    return 0;
}
