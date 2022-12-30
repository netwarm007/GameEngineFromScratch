#pragma once
#include <vector>
#include "Hitable.hpp"

namespace My {
template <class T>
class SimpleHitableList {
   public:
    __device__ SimpleHitableList(Hitable<T>** list, size_t list_size)
        : m_list(list), m_list_size(list_size) {
    }
    __device__ ~SimpleHitableList() {
        if (m_list) {
            for (int i = 0; i < m_list_size; i++) {
                delete m_list[i];
            }

            delete m_list;
        }
    }

    __device__ bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin,
                              T tmax) const {
        Hit<T> temp_hit;
        bool hit_anything = false;
        auto closest_so_far = tmax;

        for (int i = 0; i < m_list_size; i++) {
            if ((i < m_list_size) && m_list[i]->Intersect(r, temp_hit, tmin, closest_so_far)) {
                hit_anything = true;
                closest_so_far = temp_hit.getT();
                h = temp_hit;
            }
        }

        return hit_anything;
    }

    __device__ size_t size() const { return m_list_size; }

    __device__ Hitable<T>* operator[](size_t index) { return m_list[index]; }

   private:
    Hitable<T>** m_list = nullptr;
    size_t m_list_size = 0;
};

template <class T>
class HitableList : public Hitable<T> {
   public:
    HitableList() { Hitable<T>::type = HitableType::kList; }

    void add(std::shared_ptr<Hitable<T>>&& value) {
        m_Hitables.push_back(std::forward<std::shared_ptr<Hitable<T>>>(value));
    }

    constexpr auto back() { return m_Hitables.back(); }

    bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax) const override {
        Hit<T> temp_hit;
        bool hit_anything = false;
        auto closest_so_far = tmax;

        for (int i = 0; i < m_Hitables.size(); i++) {
            if (m_Hitables[i]->Intersect(r, temp_hit, tmin, closest_so_far)) {
                hit_anything = true;
                closest_so_far = temp_hit.getT();
                h = temp_hit;
            }
        }

        return hit_anything;
    }

    bool GetAabb(const Matrix4X4<T>& trans, AaBb<T, 3>& aabb) const final {
        if (m_Hitables.empty()) return false;

        AaBb<T, 3> temp_box;
        bool first_box = true;

        for (int i = 0; i < m_Hitables.size(); i++) {
            if (!m_Hitables[i]->GetAabb(trans, temp_box)) return false;
            aabb = first_box ? temp_box : SurroundingBox(aabb, temp_box);
            first_box = false;
        }

        return true;
    }

    size_t size() const { return m_Hitables.size(); }

    auto operator[](size_t index) { return m_Hitables[index]; }

    auto begin() { return m_Hitables.begin(); }

   private:
    std::ostream& dump(std::ostream& out) const override { return out; }

   private:
    std::vector<std::shared_ptr<Hitable<T>>> m_Hitables;
};
}  // namespace My
