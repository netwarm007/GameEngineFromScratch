#include <unordered_map>

#include "Bezier.hpp"
#include "Curve.hpp"
#include "ISceneParser.hpp"
#include "Linear.hpp"
#include "OpenGEX.h"
#include "SceneNode.hpp"
#include "SceneObject.hpp"
#include "portable.hpp"

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
};
}  // namespace My
