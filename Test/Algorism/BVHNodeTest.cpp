#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>

#include "BVH.hpp"

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
    std::ostream& dump(std::ostream& out) const override {
        return out << "sphere";
    }

    Vector3f center;
    float radius;
};

HitableList<float> MakeRow(int count) {
    HitableList<float> list;
    for (int i = 0; i < count; i++) {
        list.add(std::make_shared<TestSphere>(Vector3f({4.0f * i, 0, 0}), 1.0f));
    }
    return list;
}
}  // namespace

int main() {
    for (int count : {1, 2, 3, 8}) {
        auto list = MakeRow(count);
        BVHNode<float> bvh(list);

        AaBb<float, 3> box;
        CHECK(bvh.GetAabb(BuildIdentityMatrix4X4<float>(), box));
        CHECK(box.min_point()[0] == -1.0f);
        CHECK(box.max_point()[0] == 4.0f * (count - 1) + 1.0f);

        // BVH must agree with brute force over the list for every sphere.
        for (int i = 0; i < count; i++) {
            Ray<float> ray(Vector3f({4.0f * i, 0, 10}), Vector3f({0, 0, -1}));
            Hit<float> a, b;
            CHECK(bvh.Intersect(ray, a, 0.001f, 100.0f));
            CHECK(list.Intersect(ray, b, 0.001f, 100.0f));
            CHECK(std::fabs(a.getT() - b.getT()) < 1e-4f);
        }

        // From the left, the first sphere is the nearest hit.
        Hit<float> h;
        CHECK(bvh.Intersect(Ray<float>(Vector3f({-5, 0, 0}), Vector3f({1, 0, 0})),
                            h, 0.001f, 1000.0f));
        CHECK(std::fabs(h.getT() - 4.0f) < 1e-4f);

        CHECK(!bvh.Intersect(Ray<float>(Vector3f({0, 5, 10}), Vector3f({0, 0, -1})),
                             h, 0.001f, 100.0f));
        CHECK(!bvh.Intersect(Ray<float>(Vector3f({-5, 0, 0}), Vector3f({1, 0, 0})),
                             h, 0.001f, 2.0f));
    }

    BVHNode<float> bvh(MakeRow(2));
    std::ostringstream os;
    os << bvh;
    CHECK(os.str().find("sphere") != std::string::npos);

    return 0;
}
