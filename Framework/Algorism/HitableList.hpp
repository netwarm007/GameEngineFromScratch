#pragma once
#include "Hitable.hpp"

namespace My {
template <typename T>
class HitableList : public Hitable<T> {
   public:
    using value_type = std::shared_ptr<My::Hitable<T>>;
    using reference = value_type&;

    HitableList() { Hitable<T>::type = HitableType::kList; }

    template <class... Args>
    void emplace_back(Args&&... args) {
        m_Hitables.emplace_back(std::forward<Args>(args)...);
    }

    constexpr reference back() { return m_Hitables.back(); }

    bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax) const override {
        Hit<T> temp_hit;
        bool hit_anything = false;
        auto closest_so_far = tmax;

        for (const auto& hitable : m_Hitables) {
            if (hitable->Intersect(r, temp_hit, tmin, closest_so_far) &&
                hitable->Intersect(r, temp_hit, tmin, closest_so_far)) {
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

        for (const auto& hitable : m_Hitables) {
            if (!hitable->GetAabb(trans, temp_box)) return false;
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
    std::vector<value_type> m_Hitables;
};
}  // namespace My
