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
						_object = std::dynamic_pointer_cast<SceneObjectGeometry>(m_SceneObjects[_key]);
						_object->SetVisibility(_structure.GetVisibleFlag());
						_object->SetIfCastShadow(_structure.GetShadowFlag());
						_object->SetIfMotionBlur(_structure.GetMotionBlurFlag());
						const ODDL::Map<OGEX::MeshStructure> *_meshs = _structure.GetMeshMap();
						int32_t _count = _meshs->GetElementCount();
						for (int32_t i = 0; i < _count; i++)
						{
							const OGEX::MeshStructure* _mesh = (*_meshs)[i];
							SceneObjectMesh* mesh = new SceneObjectMesh();
							const std::string _primitive_type = static_cast<const char*>(_mesh->GetMeshPrimitive());
							if (_primitive_type == "points") {
								mesh->SetPrimitiveType(PrimitiveType::kPrimitiveTypePointList);
							}
							else if (_primitive_type == "lines") {
								mesh->SetPrimitiveType(PrimitiveType::kPrimitiveTypeLineList);
							}
							else if (_primitive_type == "line_strip") {
								mesh->SetPrimitiveType(PrimitiveType::kPrimitiveTypeLineStrip);
							}
							else if (_primitive_type == "triangles") {
								mesh->SetPrimitiveType(PrimitiveType::kPrimitiveTypeTriList);
							}
							else if (_primitive_type == "triangle_strip") {
								mesh->SetPrimitiveType(PrimitiveType::kPrimitiveTypeTriStrip);
							}
							else if (_primitive_type == "quads") {
								mesh->SetPrimitiveType(PrimitiveType::kPrimitiveTypeQuadList);
							}
							else {
								// not supported
								delete(mesh);
								mesh = nullptr;
							}
							if (mesh)
                            {
                                const ODDL::Structure* sub_structure = _mesh->GetFirstSubnode();
                                while (sub_structure)
                                {
                                    switch(sub_structure->GetStructureType()) 
                                    {

                                        case OGEX::kStructureVertexArray:
                                            {
                                                const OGEX::VertexArrayStructure* _v = dynamic_cast<const OGEX::VertexArrayStructure*>(sub_structure);
                                                const char* attr = _v->GetArrayAttrib();
                                                auto morph_index = _v->GetMorphIndex(); 

                                                const ODDL::Structure* _data_structure = _v->GetFirstCoreSubnode();
                                                const ODDL::DataStructure<FloatDataType>* dataStructure = dynamic_cast<const ODDL::DataStructure<FloatDataType>*>(_data_structure);

                                                auto arraySize = dataStructure->GetArraySize();
                                                auto elementCount = dataStructure->GetDataElementCount();
                                                const void* _data = &dataStructure->GetDataElement(0);
                                                void* data = new float[elementCount];
                                                size_t buf_size = sizeof(float) * elementCount;
                                                memcpy(data, _data, buf_size);
                                                VertexDataType vertexDataType;
                                                switch(arraySize) {
                                                    case 1:
                                                        vertexDataType = VertexDataType::kVertexDataTypeFloat1;
                                                        break;
                                                    case 2:
                                                        vertexDataType = VertexDataType::kVertexDataTypeFloat2;
                                                        break;
                                                    case 3:
                                                        vertexDataType = VertexDataType::kVertexDataTypeFloat3;
                                                        break;
                                                    case 4:
                                                        vertexDataType = VertexDataType::kVertexDataTypeFloat4;
                                                        break;
                                                    default:
                                                        continue;
                                                }
                                                SceneObjectVertexArray& _v_array = *new SceneObjectVertexArray(attr, morph_index, vertexDataType, data, elementCount);
                                                mesh->AddVertexArray(std::move(_v_array));
                                            }
                                            break;
                                        case OGEX::kStructureIndexArray:
                                            {
                                                const OGEX::IndexArrayStructure* _i = dynamic_cast<const OGEX::IndexArrayStructure*>(sub_structure);
                                                auto material_index = _i->GetMaterialIndex();
                                                auto restart_index = _i->GetRestartIndex();
                                                const ODDL::Structure* _data_structure = _i->GetFirstCoreSubnode();
                                                ODDL::StructureType type = _data_structure->GetStructureType();
                                                int32_t elementCount = 0;
                                                const void* _data = nullptr;
                                                IndexDataType index_type = IndexDataType::kIndexDataTypeInt16;
                                                switch(type)
                                                {
                                                    case ODDL::kDataUnsignedInt8:
                                                        {
                                                            index_type = IndexDataType::kIndexDataTypeInt8;
                                                            const ODDL::DataStructure<UnsignedInt8DataType>* dataStructure = dynamic_cast<const ODDL::DataStructure<UnsignedInt8DataType>*>(_data_structure);
                                                            elementCount = dataStructure->GetDataElementCount();
                                                            _data = &dataStructure->GetDataElement(0);

                                                        }
                                                        break;
                                                    case ODDL::kDataUnsignedInt16:
                                                        {
                                                            index_type = IndexDataType::kIndexDataTypeInt16;
                                                            const ODDL::DataStructure<UnsignedInt16DataType>* dataStructure = dynamic_cast<const ODDL::DataStructure<UnsignedInt16DataType>*>(_data_structure);
                                                            elementCount = dataStructure->GetDataElementCount();
                                                            _data = &dataStructure->GetDataElement(0);

                                                        }
                                                        break;
                                                    case ODDL::kDataUnsignedInt32:
                                                        {
                                                            index_type = IndexDataType::kIndexDataTypeInt32;
                                                            const ODDL::DataStructure<UnsignedInt32DataType>* dataStructure = dynamic_cast<const ODDL::DataStructure<UnsignedInt32DataType>*>(_data_structure);
                                                            elementCount = dataStructure->GetDataElementCount();
                                                            _data = &dataStructure->GetDataElement(0);

                                                        }
                                                        break;
                                                    case ODDL::kDataUnsignedInt64:
                                                        {
                                                            index_type = IndexDataType::kIndexDataTypeInt64;
                                                            const ODDL::DataStructure<UnsignedInt64DataType>* dataStructure = dynamic_cast<const ODDL::DataStructure<UnsignedInt64DataType>*>(_data_structure);
                                                            elementCount = dataStructure->GetDataElementCount();
                                                            _data = &dataStructure->GetDataElement(0);

                                                        }
                                                        break;
                                                    default:
                                                        ;
                                                }

                                                int32_t data_size = 0;
                                                switch(index_type)
                                                {
                                                    case IndexDataType::kIndexDataTypeInt8:
                                                        data_size = 1;
                                                        break;
                                                    case IndexDataType::kIndexDataTypeInt16:
                                                        data_size = 2;
                                                        break;
                                                    case IndexDataType::kIndexDataTypeInt32:
                                                        data_size = 3;
                                                        break;
                                                    case IndexDataType::kIndexDataTypeInt64:
                                                        data_size = 4;
                                                        break;
                                                    default:
                                                        ;
                                                }

                                                size_t buf_size = elementCount * data_size;
                                                void* data = new uint8_t[buf_size];
                                                memcpy(data, _data, buf_size);
                                                SceneObjectIndexArray& _i_array = *new SceneObjectIndexArray(material_index, restart_index, index_type, data, elementCount);
                                                mesh->AddIndexArray(std::move(_i_array));

                                            }
                                            break;
                                        default:
                                            // ignore it
                                            ;
                                    }

                                    sub_structure = sub_structure->Next();
                                }

								_object->AddMesh(std::move(*mesh));
                            }
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

