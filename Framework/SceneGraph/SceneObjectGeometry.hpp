#pragma once
#include "BaseSceneObject.hpp"
#include "SceneObjectMesh.hpp"

namespace My {
enum class SceneObjectBodyType { kAuto, kStatic, kDynamic };

// kAuto keeps the historical mapping: spheres are dynamic, other shapes static.
struct SceneObjectPhysicsProperties {
    SceneObjectBodyType bodyType{SceneObjectBodyType::kAuto};
    float mass{1.0f};
    float friction{0.5f};
    float restitution{0.0f};
    Vector3f linearVelocity{0.0f, 0.0f, 0.0f};
    // Normal axis (0=x, 1=y, 2=z) of the plane the body is confined to: it
    // cannot translate along the axis and may only spin about it. -1 means
    // unconstrained.
    int planeNormalAxis{-1};
};

class SceneObjectGeometry : public BaseSceneObject {
   protected:
    using MeshType = SceneObjectMesh;
    std::vector<std::shared_ptr<MeshType>> m_Mesh;
    bool m_bVisible;
    bool m_bShadow;
    bool m_bMotionBlur;
    SceneObjectCollisionType m_CollisionType{
        SceneObjectCollisionType::kSceneObjectCollisionTypeNone};
    float m_CollisionParameters[10];
    SceneObjectPhysicsProperties m_Physics;

   public:
    SceneObjectGeometry()
        : BaseSceneObject(SceneObjectType::kSceneObjectTypeGeometry) {}

    void SetVisibility(bool visible) { m_bVisible = visible; }
    bool Visible() { return m_bVisible; }
    void SetIfCastShadow(bool shadow) { m_bShadow = shadow; }
    bool CastShadow() { return m_bShadow; }
    void SetIfMotionBlur(bool motion_blur) { m_bMotionBlur = motion_blur; }
    bool MotionBlur() { return m_bMotionBlur; };
    void SetCollisionType(SceneObjectCollisionType collision_type) {
        m_CollisionType = collision_type;
    }
    [[nodiscard]] SceneObjectCollisionType CollisionType() const {
        return m_CollisionType;
    }
    void SetCollisionParameters(const float* param, int32_t count) {
        assert(count > 0 && count < 10);
        memcpy(m_CollisionParameters, param, sizeof(float) * count);
    }
    [[nodiscard]] const float* CollisionParameters() const {
        return m_CollisionParameters;
    }

    void SetPhysicsProperties(const SceneObjectPhysicsProperties& physics) {
        m_Physics = physics;
    }
    [[nodiscard]] const SceneObjectPhysicsProperties& PhysicsProperties()
        const {
        return m_Physics;
    }

    void AddMesh(std::shared_ptr<SceneObjectMesh>&& mesh) {
        m_Mesh.push_back(std::forward<std::shared_ptr<MeshType>>(mesh));
    }
    std::weak_ptr<MeshType> GetMesh() {
        return (m_Mesh.empty() ? nullptr : m_Mesh[0]);
    }
    [[nodiscard]] std::weak_ptr<MeshType> GetMesh() const {
        return (m_Mesh.empty() ? nullptr : m_Mesh[0]);
    }
    std::weak_ptr<MeshType> GetMeshLOD(size_t lod) {
        return (lod < m_Mesh.size() ? m_Mesh[lod] : nullptr);
    }
    [[nodiscard]] BoundingBox GetBoundingBox() const {
        return m_Mesh.empty() ? BoundingBox() : m_Mesh[0]->GetBoundingBox();
    }
    [[nodiscard]] ConvexHull<float> GetConvexHull() const {
        return m_Mesh.empty() ? ConvexHull<float>() : m_Mesh[0]->GetConvexHull();
    }

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectGeometry& obj);
};
}  // namespace My