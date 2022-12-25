#pragma once

#include <memory>
#include "random.hpp"
#include "HitableList.hpp"

namespace My {
template <class T>
class BVHNode : _implements_ Hitable<T> {
   public:
    BVHNode() { Hitable<T>::type = HitableType::kBVH; }

    BVHNode(const HitableList<T>& list) : BVHNode(list, 0, list.size()) {}

    BVHNode(const HitableList<T>& list, size_t start, size_t end) : BVHNode() {
        auto objects = list;

        int axis = random_int(0, 2);
        auto comparator = (axis == 0) ? box_compare<0>
                        : (axis == 1) ? box_compare<1>
                        : box_compare<2>;

        size_t object_span = end - start;

        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            if (comparator(objects[start], objects[start + 1])) {
                left = objects[start];
                right = objects[start + 1];
            } else {
                left = objects[start + 1];
                right = objects[start];
            }
        } else {
            std::sort(objects.begin() + start, objects.begin() + end, comparator);

            auto mid = start + object_span / 2;
            left = std::make_shared<BVHNode>(objects, start, mid);
            right = std::make_shared<BVHNode>(objects, mid, end);
        }

        AaBb<T, 3> box_left, box_right;

        if (!left->GetAabb(box_left) || !right->GetAabb(box_right)) {
            std::cerr << "No bounding box in bvh_node constructor.\n";
        }

        bounding_box = SurroundingBox(box_left, box_right);
    }

    bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax) const override {
        if (!bounding_box.Intersect(r, h, tmin, tmax)) return false;

        bool hit_left = left->Intersect(r, h, tmin, tmax);
        bool hit_right = right->Intersect(r, h, tmin, tmax);

        return hit_left || hit_right;
    }

    bool GetAabb(const Matrix4X4<T>& trans, AaBb<T, 3>& aabb) const override {
        aabb = bounding_box;
        return true;
    }

   private:
    template <int axis>
    inline static bool box_compare(const std::shared_ptr<Hitable<T>> a, const std::shared_ptr<Hitable<T>> b) {
        AaBb<T, 3> box_a;
        AaBb<T, 3> box_b;

        if (!a->GetAabb(box_a) || !b->GetAabb(box_b)) {
            std::cerr << "No bounding box in bvh_node constructor.\n";
        }

        return box_a.min()[axis] < box_b.min()[axis];
    }

    std::ostream& dump(std::ostream& out) const override {
        out << *left << std::endl;
        out << *right << std::endl;

        return out;
    }

   private:
    std::shared_ptr<Hitable<T>> left;
    std::shared_ptr<Hitable<T>> right;
    AaBb<T, 3> bounding_box;
};
}  // namespace My
