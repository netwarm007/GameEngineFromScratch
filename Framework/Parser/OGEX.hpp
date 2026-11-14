#pragma once

#include "ISceneParser.hpp"

namespace ODDL {
class Structure;
}

namespace My {
class OgexParser : _implements_ ISceneParser {
   private:
    void ConvertOddlStructureToSceneNode(
        const ODDL::Structure& structure,
        std::shared_ptr<BaseSceneNode>& base_node, Scene& scene);

   public:
    OgexParser() = default;
    virtual ~OgexParser() = default;

    std::unique_ptr<Scene> Parse(const std::string& buf) override;

   private:
    bool m_bUpIsYAxis{false};
    bool m_bInvalidCameraTarget{false};
};
}  // namespace My
