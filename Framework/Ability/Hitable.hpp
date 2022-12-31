#pragma once
#include "Ability.hpp"
#include "AaBb.hpp"
#include "Hit.hpp"
#include "Ray.hpp"
#include "portable.hpp"

namespace My {
ENUM(HitableType){kNone, kGeometry, kList, kBVH};

template <class T>
class Hitable {
   public:
    __device__ virtual ~Hitable() {}
    __device__ Hitable(HitableType type) : m_type(type) {}

    // Can intersect with a ray
    __device__ virtual bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax)
        const = 0;

    // GetAabb returns the axis aligned bounding box
    __device__ virtual bool GetAabb(AaBb<T,3>& aabb) const {
        return GetAabb(BuildIdentityMatrix4X4<T>(), aabb);
    }
    
    // GetAabb returns the axis aligned bounding box in the coordinate frame of
    // the given transform trans.
    __device__ virtual bool GetAabb(const Matrix4X4<T>& trans, AaBb<T,3>& aabb) const = 0;

    virtual void GetBoundingSphere(Vector3<T> & center, T& radius) const {
        Matrix4X4<T> tran = BuildIdentityMatrix4X4<T>();
        AaBb<T, 3> aabb;

        GetAabb(tran, aabb);

        radius = Length(aabb.max_point() - aabb.min_point()) * (T)0.5;
        center = (aabb.min_point() + aabb.max_point()) * (T)0.5;
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

        T temporalAabbMaxx = temporalAabb.max_point()[0];
        T temporalAabbMaxy = temporalAabb.max_point()[1];
        T temporalAabbMaxz = temporalAabb.max_point()[2];
        T temporalAabbMinx = temporalAabb.min_point()[0];
        T temporalAabbMiny = temporalAabb.min_point()[1];
        T temporalAabbMinz = temporalAabb.min_point()[2];

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
    friend std::ostream& operator<<(std::ostream& out, const Hitable<T>& obj) {
        out << "Hitable: "; 
        switch(obj.m_type) {
            case HitableType::kNone: out << "None"; break;
            case HitableType::kGeometry: out << "Geometry"; break;
            case HitableType::kList: out << "List"; break;
            case HitableType::kBVH: out << "BVN"; break;
        }
        out << '\t';
        obj.dump(out);

        return out;
    }

   private:
    HitableType m_type = HitableType::kNone;

   private:
    virtual std::ostream& dump(std::ostream& out) const = 0;
};

}  // namespace My
