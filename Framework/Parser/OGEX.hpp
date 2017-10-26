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
                        node = My::make_unique<SceneEmptyNode>(structure.GetStructureName());
                    }
                    break;
                case OGEX::kStructureGeometryNode:
                    {
                        node = My::make_unique<SceneGeometryNode>(structure.GetStructureName());
                    }
                    break;
                case OGEX::kStructureLightNode:
                    {
                        node = My::make_unique<SceneLightNode>(structure.GetStructureName());
                    }
                    break;
                case OGEX::kStructureCameraNode:
                    {
                        node = My::make_unique<SceneCameraNode>(structure.GetStructureName());
                    }
                    break;
                case OGEX::kStructureTransform:
                    {
                        int32_t index, count;
                        const OGEX::TransformStructure& _structure = dynamic_cast<const OGEX::TransformStructure&>(structure);
                        bool object_flag = _structure.GetObjectFlag();
                        Matrix4X4f matrix;
                        std::unique_ptr<SceneObjectTransform> transform;

                        count = _structure.GetTransformCount();
                        for (index = 0; index < count; index++) {
                            const float* data = _structure.GetTransform(index);
                            matrix = data;
                            transform = My::make_unique<SceneObjectTransform>(matrix, object_flag);
                            base_node->AppendChild(std::move(transform));
                        }
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

            base_node->AppendChild(std::move(node));
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

