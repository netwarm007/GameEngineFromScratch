#pragma once
#include "aabb.hpp"
#include "portable.hpp"

namespace My {
    ENUM(ShapeType) {
        kBox,
        kSphere,
        kCylinder,
        kCone,
        kPlane,
        kCapsule,
        kTriangle
    };

    class Shape
    {
        public:
            Shape(ShapeType shape_type) : m_kShapeType(shape_type) {};
            Shape() = delete;
            virtual ~Shape() = default;

	        // GetAabb returns the axis aligned bounding box in the coordinate frame of the given transform trans.
	        virtual void GetAabb(const Matrix4X4f& trans, 
                                    Vector3f& aabbMin, 
                                    Vector3f& aabbMax) const = 0;

	        virtual void GetBoundingSphere(Vector3f& center, float& radius) const;

	        // GetAngularMotionDisc returns the maximum radius needed for Conservative Advancement to handle 
            // time-of-impact with rotations.
	        virtual float GetAngularMotionDisc() const;

            // CalculateTemporalAabb calculates the enclosing aabb for the moving object over interval [0..timeStep)
            // result is conservative
            void CalculateTemporalAabb(const Matrix4X4f& curTrans,
                                        const Vector3f& linvel,
                                        const Vector3f& angvel,
                                        float timeStep, 
                                        Vector3f& temporalAabbMin,
                                        Vector3f& temporalAabbMax) const;

            ShapeType GetCollisionShapeType() const { return m_kShapeType; };

        protected:
            ShapeType m_kShapeType;
            float    m_fMargin;
    };
}