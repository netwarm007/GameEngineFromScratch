#pragma once
#include "BaseSceneObject.hpp"
#include "SceneObjectMesh.hpp"

namespace My {
class SceneObjectGeometry : public BaseSceneObject {
   protected:
    std::vector<std::shared_ptr<SceneObjectMesh>> m_Mesh;
    bool m_bVisible;
    bool m_bShadow;
    bool m_bMotionBlur;
    SceneObjectCollisionType m_CollisionType{
        SceneObjectCollisionType::kSceneObjectCollisionTypeNone};
    float m_CollisionParameters[10];

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

    void AddMesh(std::shared_ptr<SceneObjectMesh>&& mesh) {
        m_Mesh.push_back(std::forward<std::shared_ptr<SceneObjectMesh>>(mesh));
    }
    std::weak_ptr<SceneObjectMesh> GetMesh() {
        return (m_Mesh.empty() ? nullptr : m_Mesh[0]);
    }
    std::weak_ptr<SceneObjectMesh> GetMeshLOD(size_t lod) {
        return (lod < m_Mesh.size() ? m_Mesh[lod] : nullptr);
    }
    [[nodiscard]] BoundingBox GetBoundingBox() const {
        return m_Mesh.empty() ? BoundingBox() : m_Mesh[0]->GetBoundingBox();
    }
    [[nodiscard]] ConvexHull GetConvexHull() const {
        return m_Mesh.empty() ? ConvexHull() : m_Mesh[0]->GetConvexHull();
    }

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectGeometry& obj);
};
}  // namespace My