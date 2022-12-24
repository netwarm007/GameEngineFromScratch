#pragma once
#include "Ability.hpp"
#include "Hit.hpp"
#include "Ray.hpp"
#include "aabb.hpp"

namespace My {
template <class T>
Ability Intersectable {
   public:
    virtual ~Intersectable() = default;
    using ParamType = T;
    virtual bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax)
        const = 0;

    inline bool IntersectAabb(const Ray<T>& r, T tmin, T tmax) const {
        static const auto identity_matrix = BuildIdentityMatrix4X4<T>();

        AaBb<T, 3> aabb;
        if (GetAabb(identity_matrix, aabb)) {
            for (Dimension auto a = 0; a < 3; a++) {
                auto invD = 1.0 / r.getDirection()[a];
                auto t0 = (aabb.min()[a] - r.getOrigin()[a]) * invD;
                auto t1 = (aabb.max()[a] - r.getOrigin()[a]) * invD;
                if (invD < 0.0f) {
                    std::swap(t0, t1);
                }
                tmin = t0 > tmin ? t0 : tmin;
                tmax = t1 < tmax ? t1 : tmax;
                if (tmax <= tmin) {
                    return false;
                }
            }

            return true;
        }
        return false;
    }

    // GetAabb returns the axis aligned bounding box in the coordinate frame of
    // the given transform trans.
    virtual bool GetAabb(const Matrix4X4<T>& trans, AaBb<T, 3>& aabb) const = 0;

    virtual void GetBoundingSphere(Vector3<T> & center, T radius) const {
        Matrix4X4<T> tran;
        BuildIdentityMatrix(tran);
        AaBb<T, 3> aabb;

        GetAabb(tran, aabb);

        radius = Length(aabb.max() - aabb.min()) * 0.5;
        center = (aabb.min() + aabb.max()) * 0.5;
    }

    // GetAngularMotionDisc returns the maximum radius needed for Conservative
    // Advancement to handle
    // time-of-impact with rotations.
    [[nodiscard]] virtual float GetAngularMotionDisc() const {
        Vector3f center;
        float disc = 0.0f;
        GetBoundingSphere(center, disc);
        disc += Length(center);
        return disc;
    }

    // CalculateTemporalAabb calculates the enclosing aabb for the moving object
    // over interval [0..timeStep) result is conservative
    void CalculateTemporalAabb(
        const Matrix4X4<T>& curTrans, const Vector3<T>& linvel,
        const Vector3<T>& angvel, T timeStep, AaBb<T, 3>& temporalAabb) const {
        // start with static aabb
        GetAabb(curTrans, temporalAabb);

        T temporalAabbMaxx = temporalAabb.max()[0];
        T temporalAabbMaxy = temporalAabb.max()[1];
        T temporalAabbMaxz = temporalAabb.max()[2];
        T temporalAabbMinx = temporalAabb.min()[0];
        T temporalAabbMiny = temporalAabb.min()[1];
        T temporalAabbMinz = temporalAabb.min()[2];

        // add linear motion
        Vector3<T> linMotion = linvel * timeStep;
        ///@todo: simd would have a vector max/min operation, instead of
        /// per-element
        /// access
        if (linMotion[0] > 0.0)
            temporalAabbMaxx += linMotion[0];
        else
            temporalAabbMinx += linMotion[0];
        if (linMotion[1] > 0.0)
            temporalAabbMaxy += linMotion[1];
        else
            temporalAabbMiny += linMotion[1];
        if (linMotion[2] > 0.0)
            temporalAabbMaxz += linMotion[2];
        else
            temporalAabbMinz += linMotion[2];

        // add conservative angular motion
        T angularMotion = Length(angvel) * GetAngularMotionDisc() * timeStep;
        Vector3<T> angularMotion3d(
            {angularMotion, angularMotion, angularMotion});
        Vector3<T> temporalAabbMin =
            Vector3<T>({temporalAabbMinx, temporalAabbMiny, temporalAabbMinz});
        Vector3<T> temporalAabbMax =
            Vector3<T>({temporalAabbMaxx, temporalAabbMaxy, temporalAabbMaxz});

        temporalAabbMin = temporalAabbMin - angularMotion3d;
        temporalAabbMax = temporalAabbMax + angularMotion3d;

        temporalAabb = AaBb<T, 3>(temporalAabbMin, temporalAabbMax);
    }
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

    constexpr reference back() { return m_Intersectables.back(); }

    bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax) const override {
        Hit<T> temp_hit;
        bool hit_anything = false;
        auto closest_so_far = tmax;

        for (const auto& intersectable : m_Intersectables) {
            if (intersectable->IntersectAabb(r, tmin, closest_so_far) &&
                intersectable->Intersect(r, temp_hit, tmin, closest_so_far)) {
                hit_anything = true;
                closest_so_far = temp_hit.getT();
                h = temp_hit;
            }
        }

        return hit_anything;
    }

    bool GetAabb(const Matrix4X4<T>& trans, AaBb<T, 3>& aabb) const final {
        if (m_Intersectables.empty()) return false;

        AaBb<T, 3> temp_box;
        bool first_box = true;

        for (const auto& intersectable : m_Intersectables) {
            if (!intersectable->GetAabb(trans, temp_box)) return false;
            aabb = first_box ? temp_box : SurroundingBox(aabb, temp_box);
            first_box = false;
        }

        return true;
    }

   private:
    std::vector<value_type> m_Intersectables;
};

}  // namespace My
