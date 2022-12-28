#pragma once
#include "Hitable.hpp"

namespace My {
template <class T>
struct SimpleList {
    using value_type = T;
    using reference  = T&;
    using const_reference  = const T&;

    value_type* data;
    size_t ele_num;
    size_t max_ele_num;

    __device__ SimpleList() : data(nullptr), ele_num(0), max_ele_num(0) {}

    ~SimpleList() { delete[] data; }

    __device__ void push_back (value_type&& value) {
        if (ele_num + 1 > max_ele_num) {
            max_ele_num += 4;
            value_type* new_data = new value_type[max_ele_num];
            if (data) {
                memcpy(new_data, data, sizeof(value_type) * ele_num);
            }
            delete[] data;
            data = new_data;
        }

        data[ele_num++] = value;
    }

    __device__ constexpr reference back() {
        if (ele_num == 0) return nullptr;
        return data[ele_num - 1];
    }

    __device__ constexpr size_t size() const { return ele_num; }

    __device__ constexpr bool empty() const noexcept { return ele_num == 0; }

    __device__ constexpr reference operator[](size_t pos) { return data[pos]; }
    __device__ constexpr const_reference operator[](size_t pos) const { return data[pos]; }
};

template <class T, class ValueType = std::shared_ptr<Hitable<T>>, class ContainerType = std::vector<ValueType>>
class HitableList : public Hitable<T> {
   public:
    using value_type = ValueType;
    using reference  = ValueType&;

    __device__ HitableList() { Hitable<T>::type = HitableType::kList; }

    __device__ void add(ValueType&& value) {
        m_Hitables.push_back(std::forward<ValueType>(value));
    }

    constexpr reference back() { return m_Hitables.back(); }

    __device__ bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax) const override {
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

    __device__ bool GetAabb(const Matrix4X4<T>& trans, AaBb<T, 3>& aabb) const final {
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

    size_t size() const {
        return m_Hitables.size();
    }

    auto operator[](size_t index) {
        return m_Hitables[index];
    }

    auto begin() {
        return m_Hitables.begin();
    }

   private:
    std::ostream& dump(std::ostream& out) const override {
        return out;
    }

   private:
    ContainerType m_Hitables;
};
}  // namespace My
