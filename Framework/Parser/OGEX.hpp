#include <unordered_map>
#include "OpenGEX.h"
#include "portable.hpp"
#include "SceneParser.hpp"

namespace My {
    class OgexParser : implements SceneParser
    {
    private:
        std::unordered_map<std::string, std::shared_ptr<BaseSceneObject>> m_SceneObjects;

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
						SceneGeometryNode& _node = dynamic_cast<SceneGeometryNode&>(*node);
						const OGEX::GeometryNodeStructure& _structure = dynamic_cast<const OGEX::GeometryNodeStructure&>(structure);
                        std::string _key = _structure.GetObjectStructure()->GetStructureName();
						_node.SetVisibility(_structure.GetVisibleFlag());
						_node.SetIfCastShadow(_structure.GetShadowFlag());
						_node.SetIfMotionBlur(_structure.GetMotionBlurFlag());
                        if(!m_SceneObjects[_key]) {
							m_SceneObjects[_key] = std::make_shared<SceneObjectGeometry>();
                        }
                        _node.AddSceneObjectRef(std::dynamic_pointer_cast<SceneObjectGeometry>(m_SceneObjects[_key]));
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
                case OGEX::kStructureGeometryObject:
                    {
						const OGEX::GeometryObjectStructure& _structure = dynamic_cast<const OGEX::GeometryObjectStructure&>(structure);
                        std::string _key = _structure.GetStructureName();
						std::shared_ptr<SceneObjectGeometry> _object;
                        if(!m_SceneObjects[_key]) {
							m_SceneObjects[_key] = std::make_shared<SceneObjectGeometry>();
						}
						else {
							_object = std::dynamic_pointer_cast<SceneObjectGeometry>(m_SceneObjects[_key]);
						}
						_object->SetVisibility(_structure.GetVisibleFlag());
						_object->SetIfCastShadow(_structure.GetShadowFlag());
						_object->SetIfMotionBlur(_structure.GetMotionBlurFlag());
						const ODDL::Map<OGEX::MeshStructure> *_meshs = _structure.GetMeshMap();
						int32_t _count = _meshs->GetElementCount();
						for (int32_t i = 0; i < _count; i++)
						{
							const OGEX::MeshStructure* _mesh = (*_meshs)[i];
							SceneObjectMesh* mesh = new SceneObjectMesh();
							const std::string _primitive_type = _mesh->GetMeshPrimitive();
							if (_primitive_type == "points") {
								mesh->SetPrimitiveType(kPrimitiveTypePointList);
							}
							else if (_primitive_type == "lines") {
								mesh->SetPrimitiveType(kPrimitiveTypeLineList);
							}
							else if (_primitive_type == "line_strip") {
								mesh->SetPrimitiveType(kPrimitiveTypeLineStrip);
							}
							else if (_primitive_type == "triangles") {
								mesh->SetPrimitiveType(kPrimitiveTypeTriList);
							}
							else if (_primitive_type == "triangle_strip") {
								mesh->SetPrimitiveType(kPrimitiveTypeTriStrip);
							}
							else if (_primitive_type == "quads") {
								mesh->SetPrimitiveType(kPrimitiveTypeQuadList);
							}
							else {
								// not supported
								delete(mesh);
								mesh = nullptr;
							}
							if (mesh)
								_object->AddMesh(std::move(mesh));
						}
                    }
                    return;
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
				case OGEX::kStructureMesh:
					{
						const OGEX::MeshStructure& _structure = dynamic_cast<const OGEX::MeshStructure&>(structure);
					}
					return;
				case OGEX::kStructureVertexArray:
					{
					}
					return;
				case OGEX::kStructureIndexArray:
					{
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

