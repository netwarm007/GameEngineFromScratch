#include "Geometry.hpp"

using namespace My;

void Geometry::CalculateTemporalAabb(const Matrix4X4f& curTrans,
                                        const Vector3f& linvel,
                                        const Vector3f& angvel,
                                        float timeStep, 
                                        Vector3f& temporalAabbMin,
                                        Vector3f& temporalAabbMax) const
{
	//start with static aabb
	GetAabb(curTrans,temporalAabbMin,temporalAabbMax);

	float temporalAabbMaxx = temporalAabbMax.x;
	float temporalAabbMaxy = temporalAabbMax.y;
	float temporalAabbMaxz = temporalAabbMax.z;
	float temporalAabbMinx = temporalAabbMin.x;
	float temporalAabbMiny = temporalAabbMin.y;
	float temporalAabbMinz = temporalAabbMin.z;

	// add linear motion
	Vector3f linMotion = linvel * timeStep;
	///@todo: simd would have a vector max/min operation, instead of per-element access
	if (linMotion.x > 0.0f)
		temporalAabbMaxx += linMotion.x; 
	else
		temporalAabbMinx += linMotion.x;
	if (linMotion.y > 0.0f)
		temporalAabbMaxy += linMotion.y; 
	else
		temporalAabbMiny += linMotion.y;
	if (linMotion.z > 0.0f)
		temporalAabbMaxz += linMotion.z; 
	else
		temporalAabbMinz += linMotion.z;

	//add conservative angular motion
	float angularMotion = Length(angvel) * GetAngularMotionDisc() * timeStep;
	Vector3f angularMotion3d(angularMotion,angularMotion,angularMotion);
	temporalAabbMin = Vector3f(temporalAabbMinx,temporalAabbMiny,temporalAabbMinz);
	temporalAabbMax = Vector3f(temporalAabbMaxx,temporalAabbMaxy,temporalAabbMaxz);

	temporalAabbMin = temporalAabbMin - angularMotion3d;
	temporalAabbMax = temporalAabbMax + angularMotion3d;
}

void Geometry::GetBoundingSphere(Vector3f& center, float& radius) const
{
	Matrix4X4f tran;
	BuildIdentityMatrix(tran);
	Vector3f aabbMin,aabbMax;

	GetAabb(tran, aabbMin, aabbMax);

	radius = Length(aabbMax - aabbMin) * 0.5f;
	center = (aabbMin + aabbMax) * 0.5f;
}

float Geometry::GetAngularMotionDisc() const
{
	Vector3f    center;
	float       disc = 0.0f;
	GetBoundingSphere(center, disc);
	disc += Length(center);
	return disc;
}