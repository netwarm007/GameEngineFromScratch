#pragma once

#include <memory>
#ifdef __CUDACC__
#include <curand_kernel.h>
#include <thrust/sort.h>
#endif
#include "HitableList.hpp"
#include "random.hpp"

namespace My {
#ifdef __CUDACC__

template <class T>
class SimpleBVHNode : _implements_ Hitable<T> {
   public:
    __device__ SimpleBVHNode() : Hitable<T>(HitableType::kBVH) {}
    __device__ SimpleBVHNode(Hitable<T>** list, int start, int end,
                             curandStateMRG32k3a_t* local_rand_state)
        : Hitable<T>(HitableType::kBVH) {
        struct stack_element_type {
            SimpleBVHNode* root;
            int start;
            int end;
        };

        //printf("Constructing BVH\n");

        int stack_length = logf(end - start) / logf(2.0f) * 2.0f + 1.0f;

        stack_element_type* pStack = new stack_element_type[stack_length];
        stack_element_type* pStackTop = pStack;

        pStackTop->root = this;
        pStackTop->start = start;
        pStackTop->end = end;

        pStackTop++;

        while (pStackTop > pStack) {
            //printf("layer %d\n", (int)(pStackTop - pStack));
            pStackTop--;

            SimpleBVHNode* cur_node = pStackTop->root;

            if (cur_node->left && cur_node->right) {
                AaBb<T, 3> box_left;
                AaBb<T, 3> box_right;

                if (!cur_node->left->GetAabb(box_left) ||
                    !cur_node->right->GetAabb(box_right)) {
                    printf("No bounding box in bvh_node constructor.\n");
                }

                cur_node->bounding_box = SurroundingBox(box_left, box_right);
                //printf(
                //    "Constructed Bounding Box: {(%f, %f, %f) - (%f, %f, %f)}\n",
                //    cur_node->bounding_box.min_point()[0],
                //    cur_node->bounding_box.min_point()[1],
                //    cur_node->bounding_box.min_point()[2],
                //    cur_node->bounding_box.max_point()[0],
                //    cur_node->bounding_box.max_point()[1],
                //    cur_node->bounding_box.max_point()[2]);

                    continue;
            }

            int axis = random_int(0, 2, local_rand_state);
            auto comparator = (axis == 0)   ? box_compare<0>
                              : (axis == 1) ? box_compare<1>
                                            : box_compare<2>;
            int object_span = pStackTop->end - pStackTop->start;

            if (object_span == 1) {
                //printf("leaf node type 1\n");
                pStackTop->root->left = pStackTop->root->right =
                    list[pStackTop->start];

                pStackTop++;  // to enter the bounding box calculation.
            } else if (object_span == 2) {
                if (comparator(list[pStackTop->start],
                               list[pStackTop->start + 1])) {
                    //printf("leaf node type 2\n");
                    pStackTop->root->left = list[pStackTop->start];
                    pStackTop->root->right = list[pStackTop->start + 1];
                } else {
                    //printf("leaf node type 3\n");
                    pStackTop->root->left = list[pStackTop->start + 1];
                    pStackTop->root->right = list[pStackTop->start];
                }

                pStackTop++;  // to enter the bounding box calculation.
            } else {
                //printf("non-leaf node.\n");
                //printf("sort against axis %d\n", axis);
                thrust::sort(&list[pStackTop->start], &list[pStackTop->end],
                             comparator);

                int start = pStackTop->start;
                int end = pStackTop->end;
                int mid = pStackTop->start + (object_span >> 1);
                //printf("start = %d, mid = %d, end = %d, object_span = %d\n",
                //       pStackTop->start, mid, pStackTop->end, object_span);

                // push self
                pStackTop++;

                // push right
                //printf("push right: [%d %d)\n", mid, end);
                pStackTop->root = new SimpleBVHNode();
                pStackTop->start = mid;
                pStackTop->end = end;
                cur_node->right = pStackTop->root;

                pStackTop++;
                assert(pStackTop - pStack <= stack_length);

                // push left
                //printf("push left: [%d %d)\n", start, mid);
                pStackTop->root = new SimpleBVHNode();
                pStackTop->start = start;
                pStackTop->end = mid;
                cur_node->left = pStackTop->root;

                pStackTop++;
                assert(pStackTop - pStack <= stack_length);
            }
        }

        delete pStack;
    }

    __device__ virtual ~SimpleBVHNode() {
        delete left;
        if (right != left) delete right; // avoid double free
    }

    __device__ bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin,
                              T tmax) const override {
        if (!bounding_box.Intersect(r, tmin, tmax)) {
            return false;
        }

        bool hit_left = left->Intersect(r, h, tmin, tmax);
        bool hit_right =
            right->Intersect(r, h, tmin, hit_left ? h.getT() : tmax);

        return hit_left || hit_right;
    }

    __device__ bool GetAabb(AaBb<T, 3>& aabb) const override {
        aabb = bounding_box;
        return true;
    }

    __device__ bool GetAabb(const Matrix4X4<T>& trans,
                            AaBb<T, 3>& aabb) const override {
        aabb = bounding_box;
        return true;
    }

   private:
    std::ostream& dump(std::ostream& out) const override { return out; }

    template <int axis>
    __device__ inline static bool box_compare(const Hitable<T>* a,
                                              const Hitable<T>* b) {
        AaBb<T, 3> box_a;
        AaBb<T, 3> box_b;

        if (!a->GetAabb(box_a) || !b->GetAabb(box_b)) {
            printf("No bounding box in bvh_node constructor.\n");
        }

        return box_a.min_point()[axis] < box_b.min_point()[axis];
    }

   private:
    Hitable<T>* left = nullptr;
    Hitable<T>* right = nullptr;
    AaBb<T, 3> bounding_box;
};

#else

template <class T>
class BVHNode : _implements_ Hitable<T> {
   public:
    BVHNode() : Hitable<T>(HitableType::kBVH) {}

    BVHNode(const HitableList<T>& list) : BVHNode(list, 0, list.size()) {}

    BVHNode(const HitableList<T>& list, size_t start, size_t end) : BVHNode() {
        auto objects = list;

        int axis = random_int(0, 2);
        auto comparator = (axis == 0)   ? box_compare<0>
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
            std::sort(objects.begin() + start, objects.begin() + end,
                      comparator);

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
        if (!bounding_box.Intersect(r, tmin, tmax)) return false;

        bool hit_left = left->Intersect(r, h, tmin, tmax);
        bool hit_right =
            right->Intersect(r, h, tmin, hit_left ? h.getT() : tmax);

        return hit_left || hit_right;
    }

    bool GetAabb(const Matrix4X4<T>& trans, AaBb<T, 3>& aabb) const override {
        aabb = bounding_box;
        return true;
    }

   private:
    template <int axis>
    inline static bool box_compare(const std::shared_ptr<Hitable<T> > a,
                                   const std::shared_ptr<Hitable<T> > b) {
        AaBb<T, 3> box_a;
        AaBb<T, 3> box_b;

        if (!a->GetAabb(box_a) || !b->GetAabb(box_b)) {
            std::cerr << "No bounding box in bvh_node constructor.\n";
        }

        return box_a.min_point()[axis] < box_b.min_point()[axis];
    }

    std::ostream& dump(std::ostream& out) const override {
        out << *left << std::endl;
        out << *right << std::endl;

        return out;
    }

   private:
    std::shared_ptr<Hitable<T> > left;
    std::shared_ptr<Hitable<T> > right;
    AaBb<T, 3> bounding_box;
};

#endif  // CUDACC
}  // namespace My
