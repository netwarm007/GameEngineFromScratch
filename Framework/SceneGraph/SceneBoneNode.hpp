#pragma once
#include "BaseSceneNode.hpp"

namespace My {
class SceneBoneNode : public BaseSceneNode {
    using BaseSceneNode::BaseSceneNode;

   public:
    [[nodiscard]] SceneNodeKind GetKind() const override {
        return SceneNodeKind::Bone;
    }
};
}  // namespace My