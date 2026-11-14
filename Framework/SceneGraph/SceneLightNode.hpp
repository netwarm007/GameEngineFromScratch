#pragma once
#include "BaseSceneNode.hpp"

namespace My {
class SceneLightNode : public SceneNode<SceneObjectLight> {
   protected:
    bool m_bShadow;

   public:
    using SceneNode::SceneNode;

    [[nodiscard]] SceneNodeKind GetKind() const override {
        return SceneNodeKind::Light;
    }

    void SetIfCastShadow(bool shadow) { m_bShadow = shadow; };
    bool CastShadow() const { return m_bShadow; };
};
}  // namespace My