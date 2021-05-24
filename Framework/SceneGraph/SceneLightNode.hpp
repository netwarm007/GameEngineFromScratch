#pragma once
#include "BaseSceneNode.hpp"

namespace My {
class SceneLightNode : public SceneNode<SceneObjectLight> {
   protected:
    bool m_bShadow;

   public:
    using SceneNode::SceneNode;

    void SetIfCastShadow(bool shadow) { m_bShadow = shadow; };
    bool CastShadow() { return m_bShadow; };
};
}  // namespace My