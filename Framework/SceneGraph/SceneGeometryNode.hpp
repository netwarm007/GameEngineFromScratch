#pragma once
#include <optional>

#include "BaseSceneNode.hpp"

namespace My {
class SceneGeometryNode : public SceneNode<SceneObjectGeometry> {
    using SceneNodeType = SceneNode<SceneObjectGeometry>;
   protected:
    bool m_bVisible;
    bool m_bShadow;
    bool m_bMotionBlur;
    std::vector<std::string> m_Materials;
    std::optional<Matrix4X4f> m_SimulationTransform;

   protected:
    void dump(std::ostream& out) const override {
        SceneNodeType::dump(out);
        out << "Visible: " << m_bVisible << std::endl;
        out << "Shadow: " << m_bShadow << std::endl;
        out << "Motion Blur: " << m_bMotionBlur << std::endl;
        out << "Material(s): " << std::endl;
        for (const auto& material : m_Materials) {
            out << material << std::endl;
        }
    };

   public:
    using SceneNodeType::SceneNode;

    [[nodiscard]] SceneNodeKind GetKind() const override {
        return SceneNodeKind::Geometry;
    }

    void SetVisibility(bool visible) { m_bVisible = visible; };
    bool Visible() const { return m_bVisible; };
    void SetIfCastShadow(bool shadow) { m_bShadow = shadow; };
    bool CastShadow() const { return m_bShadow; };
    void SetIfMotionBlur(bool motion_blur) { m_bMotionBlur = motion_blur; };
    bool MotionBlur() const { return m_bMotionBlur; };
    using SceneNodeType::AddSceneObjectRef;
    void AddMaterialRef(const std::string& key) { m_Materials.push_back(key); };
    void AddMaterialRef(const std::string&& key) {
        m_Materials.push_back(key);
    };
    std::string GetMaterialRef(const size_t index) const {
        if (index < m_Materials.size()) {
            return m_Materials[index];
        }

        return std::string("default");
    };

    void SetSimulationTransform(const Matrix4X4f& transform) {
        m_SimulationTransform = transform;
    }

    void ClearSimulationTransform() { m_SimulationTransform.reset(); }

    [[nodiscard]] Matrix4X4f GetRenderTransform() const {
        if (m_SimulationTransform) {
            return *m_SimulationTransform;
        }
        return *GetCalculatedTransform();
    }
};
}  // namespace My