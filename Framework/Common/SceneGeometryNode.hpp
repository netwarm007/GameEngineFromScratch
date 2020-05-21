#pragma once
#include "BaseSceneNode.hpp"

namespace My {
class SceneGeometryNode : public SceneNode<SceneObjectGeometry> {
   protected:
    bool m_bVisible;
    bool m_bShadow;
    bool m_bMotionBlur;
    std::vector<std::string> m_Materials;
    void* m_pRigidBody = nullptr;

   protected:
    void dump(std::ostream& out) const override {
        SceneNode::dump(out);
        out << "Visible: " << m_bVisible << std::endl;
        out << "Shadow: " << m_bShadow << std::endl;
        out << "Motion Blur: " << m_bMotionBlur << std::endl;
        out << "Material(s): " << std::endl;
        for (const auto& material : m_Materials) {
            out << material << std::endl;
        }
    };

   public:
    using SceneNode::SceneNode;

    void SetVisibility(bool visible) { m_bVisible = visible; };
    bool Visible() { return m_bVisible; };
    void SetIfCastShadow(bool shadow) { m_bShadow = shadow; };
    bool CastShadow() { return m_bShadow; };
    void SetIfMotionBlur(bool motion_blur) { m_bMotionBlur = motion_blur; };
    bool MotionBlur() { return m_bMotionBlur; };
    using SceneNode::AddSceneObjectRef;
    void AddMaterialRef(const std::string& key) { m_Materials.push_back(key); };
    void AddMaterialRef(const std::string&& key) {
        m_Materials.push_back(key);
    };
    std::string GetMaterialRef(const size_t index) {
        if (index < m_Materials.size()) {
            return m_Materials[index];
        }

        return std::string("default");
    };

    void LinkRigidBody(void* rigidBody) { m_pRigidBody = rigidBody; }

    void* UnlinkRigidBody() {
        void* rigidBody = m_pRigidBody;
        m_pRigidBody = nullptr;

        return rigidBody;
    }

    void* RigidBody() { return m_pRigidBody; }
};
}  // namespace My