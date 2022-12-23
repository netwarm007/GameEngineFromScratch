#pragma once
#include "Ability.hpp"
#include "Ray.hpp"
#include "Hit.hpp"

namespace My {
template <typename T>
Ability Intersectable {
   public:
    virtual ~Intersectable() = default;
    using ParamType = const T;
    virtual bool Intersect(const Ray<T> &r, Hit<T> &h, T tmin, T tmax) const = 0;
};

template <typename T>
class IntersectableList : _implements_ My::Intersectable<T> {
public:
    using value_type = std::shared_ptr<My::Intersectable<T>>;
    using reference = value_type&;

    template <class... Args>
    void emplace_back(Args&&... args) {
        m_Intersectables.emplace_back(std::forward<Args>(args)...);
    }

    constexpr reference back() {
        return m_Intersectables.back();
    }

    bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax) const override {
        Hit<T> temp_hit;
        bool hit_anything = false;
        auto closest_so_far = tmax;

        for (const auto& intersectable : m_Intersectables) {
            if (intersectable->Intersect(r, temp_hit, tmin, closest_so_far)) {
                hit_anything = true;
                closest_so_far = temp_hit.getT();
                h = temp_hit;
            }
        }

        return hit_anything;
    }

private:
    std::vector<value_type> m_Intersectables;
};

}  // namespace My
