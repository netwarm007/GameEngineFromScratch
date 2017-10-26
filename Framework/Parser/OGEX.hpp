#include "OpenGEX.h"
#include "portable.hpp"
#include "SceneParser.hpp"

namespace My {
    class OgexParser : implements SceneParser
    {
    private:
        void ConvertOddlStructureToSceneNode(const ODDL::Structure& structure, std::unique_ptr<BaseSceneNode>& base_node)
        {
            std::unique_ptr<BaseSceneNode> node;

            switch(structure.GetStructureType()) {
                case OGEX::kStructureNode:
                    {
                        node = std::make_unique<SceneEmptyNode>(structure.GetStructureName());
                        base_node->AppendChild(std::move(node));
                    }
                    break;
                case OGEX::kStructureGeometryNode:
                    {
                        node = std::make_unique<SceneGeometryNode>(structure.GetStructureName());
                        base_node->AppendChild(std::move(node));
                    }
                    break;
                case OGEX::kStructureLightNode:
                    {
                        node = std::make_unique<SceneLightNode>(structure.GetStructureName());
                        base_node->AppendChild(std::move(node));
                    }
                    break;
                case OGEX::kStructureCameraNode:
                    {
                        node = std::make_unique<SceneCameraNode>(structure.GetStructureName());
                        base_node->AppendChild(std::move(node));
                    }
                    break;
                case OGEX::kStructureTransform:
                    {
                        std::unique_ptr<SceneObjectTransform> transform (new SceneObjectTransform());
                        base_node->AppendChild(std::move(transform));
                    }
                    return;
                default:
                    // just ignore it and finish
                    return;
            };

            const ODDL::Structure* sub_structure = structure.GetFirstSubnode();
            while (sub_structure)
            {
                ConvertOddlStructureToSceneNode(*sub_structure, node);

                sub_structure = sub_structure->Next();
            }
        }

    public:
        virtual std::unique_ptr<BaseSceneNode> Parse(const std::string& buf)
        {
            std::unique_ptr<BaseSceneNode> root_node (new BaseSceneNode("scene_root"));
            OGEX::OpenGexDataDescription  openGexDataDescription;

            ODDL::DataResult result = openGexDataDescription.ProcessText(buf.c_str());
            if (result == ODDL::kDataOkay)
            {
                const ODDL::Structure* structure = openGexDataDescription.GetRootStructure()->GetFirstSubnode();
                while (structure)
                {
                    ConvertOddlStructureToSceneNode(*structure, root_node);

                    structure = structure->Next();
                }
            }

            return std::move(root_node);
        }
    };
}

