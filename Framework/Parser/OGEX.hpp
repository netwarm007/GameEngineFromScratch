#include <unordered_map>
#include "OpenGEX.h"
#include "portable.hpp"
#include "SceneParser.hpp"

namespace My {
    class OgexParser : implements SceneParser
    {
    private:
        void ConvertOddlStructureToSceneNode(const ODDL::Structure& structure, std::shared_ptr<BaseSceneNode>& base_node, Scene& scene)
        {
            std::shared_ptr<BaseSceneNode> node;

            switch(structure.GetStructureType()) {
                case OGEX::kStructureMetric:
                    {
                        const OGEX::MetricStructure& _structure = dynamic_cast<const OGEX::MetricStructure&>(structure);
                        auto _key = _structure.GetMetricKey();
                        const ODDL::Structure *sub_structure = _structure.GetFirstCoreSubnode();
                        if (_key == "up") {
		                    const ODDL::DataStructure<ODDL::StringDataType> *dataStructure = static_cast<const ODDL::DataStructure<ODDL::StringDataType> *>(sub_structure);
                            auto axis_name = dataStructure->GetDataElement(0);
                            if (axis_name == "y") {
                                m_bUpIsYAxis = true;
                            }
                            else {
                                m_bUpIsYAxis = false;
                            }
                        }
                    }
                    return;
                case OGEX::kStructureNode:
                    {
                        node = std::make_shared<SceneEmptyNode>(structure.GetStructureName());
                    }
                    break;
                case OGEX::kStructureGeometryNode:
                    {
                        auto _node = std::make_shared<SceneGeometryNode>(structure.GetStructureName());
						const OGEX::GeometryNodeStructure& _structure = dynamic_cast<const OGEX::GeometryNodeStructure&>(structure);

						_node->SetVisibility(_structure.GetVisibleFlag());
						_node->SetIfCastShadow(_structure.GetShadowFlag());
						_node->SetIfMotionBlur(_structure.GetMotionBlurFlag());

                        // ref scene objects
                        std::string _key = _structure.GetObjectStructure()->GetStructureName();
                        _node->AddSceneObjectRef(_key);

                        // ref materials
                        auto materials = _structure.GetMaterialStructureArray();
                        auto materials_count = materials.GetElementCount();
                        for (auto i = 0; i < materials_count; i++)
                        {
                            auto material = materials[i];
                            _key = material->GetStructureName();
                            _node->AddMaterialRef(_key);
                        }

                        scene.GeometryNodes.emplace(_key, _node);

                        node = _node;
                    }
                    break;
                case OGEX::kStructureLightNode:
                    {
                        auto _node = std::make_shared<SceneLightNode>(structure.GetStructureName());
						const OGEX::LightNodeStructure& _structure = dynamic_cast<const OGEX::LightNodeStructure&>(structure);

                        _node->SetIfCastShadow(_structure.GetShadowFlag());

                        // ref scene objects
                        std::string _key = _structure.GetObjectStructure()->GetStructureName();
                        _node->AddSceneObjectRef(_key);

                        scene.LightNodes.emplace(_key, _node);

                        node = _node;
                    }
                    break;
                case OGEX::kStructureCameraNode:
                    {
                        auto _node = std::make_shared<SceneCameraNode>(structure.GetStructureName());
						const OGEX::CameraNodeStructure& _structure = dynamic_cast<const OGEX::CameraNodeStructure&>(structure);

                        // ref scene objects
                        std::string _key = _structure.GetObjectStructure()->GetStructureName();
                        _node->AddSceneObjectRef(_key);

                        scene.CameraNodes.emplace(_key, _node);

                        node = _node;
                    }
                    break;
                case OGEX::kStructureGeometryObject:
                    {
						const OGEX::GeometryObjectStructure& _structure = dynamic_cast<const OGEX::GeometryObjectStructure&>(structure);
                        std::string _key = _structure.GetStructureName();
						auto _object = std::make_shared<SceneObjectGeometry>();
						_object->SetVisibility(_structure.GetVisibleFlag());
						_object->SetIfCastShadow(_structure.GetShadowFlag());
						_object->SetIfMotionBlur(_structure.GetMotionBlurFlag());
						const ODDL::Map<OGEX::MeshStructure> *_meshs = _structure.GetMeshMap();
						int32_t _count = _meshs->GetElementCount();
						for (int32_t i = 0; i < _count; i++)
						{
							const OGEX::MeshStructure* _mesh = (*_meshs)[i];
                            std::shared_ptr<SceneObjectMesh> mesh(new SceneObjectMesh());
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
								mesh.reset();
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
                                                        data_size = 4;
                                                        break;
                                                    case IndexDataType::kIndexDataTypeInt64:
                                                        data_size = 8;
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

								_object->AddMesh(mesh);
                            }
						}

                        scene.Geometries[_key] = _object;
                    }
                    return;
                case OGEX::kStructureTransform:
                    {
                        int32_t index, count;
                        const OGEX::TransformStructure& _structure = dynamic_cast<const OGEX::TransformStructure&>(structure);
                        bool object_flag = _structure.GetObjectFlag();
                        Matrix4X4f matrix;
                        std::shared_ptr<SceneObjectTransform> transform;

                        count = _structure.GetTransformCount();
                        for (index = 0; index < count; index++) {
                            const float* data = _structure.GetTransform(index);
                            matrix = data;
                            if (!m_bUpIsYAxis) {
                                // exchange y and z
                                // ExchangeYandZ(matrix);
                            }
                            transform = std::make_shared<SceneObjectTransform>(matrix, object_flag);
                            base_node->AppendChild(std::move(transform));
                        }
                    }
                    return;
                case OGEX::kStructureMaterial:
                    {
                        const OGEX::MaterialStructure& _structure = dynamic_cast<const OGEX::MaterialStructure&>(structure);
                        std::string material_name;
                        const char* _name = _structure.GetMaterialName();
                        std::string _key = _structure.GetStructureName();
                        auto material = std::make_shared<SceneObjectMaterial>();
                        material->SetName(_name);

                        const ODDL::Structure* _sub_structure = _structure.GetFirstCoreSubnode();
                        while(_sub_structure) {
                            std::string attrib, textureName;
                            Vector4f color;
                            float param;
                            switch(_sub_structure->GetStructureType())
                            {
                                case OGEX::kStructureColor:
                                    {
				                        attrib = dynamic_cast<const OGEX::ColorStructure*>(_sub_structure)->GetAttribString();
				                        color = dynamic_cast<const OGEX::ColorStructure*>(_sub_structure)->GetColor();
                                        material->SetColor(attrib, color);
                                    }
                                    break;
                                case OGEX::kStructureParam:
                                    {
				                        attrib = dynamic_cast<const OGEX::ParamStructure*>(_sub_structure)->GetAttribString();
				                        param = dynamic_cast<const OGEX::ParamStructure*>(_sub_structure)->GetParam();
                                        material->SetParam(attrib, param);
                                    }
                                    break;
                                case OGEX::kStructureTexture:
                                    {
				                        attrib = dynamic_cast<const OGEX::TextureStructure*>(_sub_structure)->GetAttribString();
				                        textureName = dynamic_cast<const OGEX::TextureStructure*>(_sub_structure)->GetTextureName();
                                        material->SetTexture(attrib, textureName);
                                    }
                                    break;
                                default:
                                    ;
                            };
                            
                            _sub_structure = _sub_structure->Next();
                        }
                        scene.Materials[_key] = material;
                    }
                    return;
                case OGEX::kStructureLightObject:
                    {
                        const OGEX::LightObjectStructure& _structure = dynamic_cast<const OGEX::LightObjectStructure&>(structure);
                        const char* _type_str = _structure.GetTypeString();
                        const bool _bshadow = _structure.GetShadowFlag();
                        std::string _key = _structure.GetStructureName();
                        std::shared_ptr<SceneObjectLight> light;

                        if (!strncmp(_type_str, "infinite", 8))
                        {
                            light = std::make_shared<SceneObjectInfiniteLight>();
                        }
                        else if (!strncmp(_type_str, "point", 5))
                        {
                            light = std::make_shared<SceneObjectOmniLight>();
                        }
                        else if (!strncmp(_type_str, "spot", 5))
                        {
                            light = std::make_shared<SceneObjectSpotLight>();
                        }

                        light->SetIfCastShadow(_bshadow);

                        const ODDL::Structure* _sub_structure = _structure.GetFirstCoreSubnode();
                        while(_sub_structure) {
                            std::string attrib, textureName;
                            Vector4f color;
                            float param;
                            switch(_sub_structure->GetStructureType())
                            {
                                case OGEX::kStructureColor:
                                    {
				                        attrib = dynamic_cast<const OGEX::ColorStructure*>(_sub_structure)->GetAttribString();
				                        color = dynamic_cast<const OGEX::ColorStructure*>(_sub_structure)->GetColor();
                                        light->SetColor(attrib, color);
                                    }
                                    break;
                                case OGEX::kStructureParam:
                                    {
				                        attrib = dynamic_cast<const OGEX::ParamStructure*>(_sub_structure)->GetAttribString();
				                        param = dynamic_cast<const OGEX::ParamStructure*>(_sub_structure)->GetParam();
                                        light->SetParam(attrib, param);
                                    }
                                    break;
                                case OGEX::kStructureTexture:
                                    {
				                        attrib = dynamic_cast<const OGEX::TextureStructure*>(_sub_structure)->GetAttribString();
				                        textureName = dynamic_cast<const OGEX::TextureStructure*>(_sub_structure)->GetTextureName();
                                        light->SetTexture(attrib, textureName);
                                    }
                                    break;
                                case OGEX::kStructureAtten:
                                    {
                                        // TODO: truly implement it
                                        light->SetAttenuation(DefaultAttenFunc);
                                    }
                                    break;
                                default:
                                    ;
                            };
                            
                            _sub_structure = _sub_structure->Next();
                        }
                        scene.Lights[_key] = light;
                    }
                    return;
                case OGEX::kStructureCameraObject:
                    {
                        const OGEX::CameraObjectStructure& _structure = dynamic_cast<const OGEX::CameraObjectStructure&>(structure);
                        std::string _key = _structure.GetStructureName();
                        auto camera = std::make_shared<SceneObjectPerspectiveCamera>();

                        const ODDL::Structure* _sub_structure = _structure.GetFirstCoreSubnode();
                        while(_sub_structure) {
                            std::string attrib, textureName;
                            Vector4f color;
                            float param;
                            switch(_sub_structure->GetStructureType())
                            {
                                case OGEX::kStructureColor:
                                    {
				                        attrib = dynamic_cast<const OGEX::ColorStructure*>(_sub_structure)->GetAttribString();
				                        color = dynamic_cast<const OGEX::ColorStructure*>(_sub_structure)->GetColor();
                                        camera->SetColor(attrib, color);
                                    }
                                    break;
                                case OGEX::kStructureParam:
                                    {
				                        attrib = dynamic_cast<const OGEX::ParamStructure*>(_sub_structure)->GetAttribString();
				                        param = dynamic_cast<const OGEX::ParamStructure*>(_sub_structure)->GetParam();
                                        camera->SetParam(attrib, param);
                                    }
                                    break;
                                case OGEX::kStructureTexture:
                                    {
				                        attrib = dynamic_cast<const OGEX::TextureStructure*>(_sub_structure)->GetAttribString();
				                        textureName = dynamic_cast<const OGEX::TextureStructure*>(_sub_structure)->GetTextureName();
                                        camera->SetTexture(attrib, textureName);
                                    }
                                    break;
                                default:
                                    ;
                            };
                            
                            _sub_structure = _sub_structure->Next();
                        }
                        scene.Cameras[_key] = camera;
                    }
                    return;
                default:
                    // just ignore it and finish
                    return;
            };

            const ODDL::Structure* sub_structure = structure.GetFirstSubnode();
            while (sub_structure)
            {
                ConvertOddlStructureToSceneNode(*sub_structure, node, scene);

                sub_structure = sub_structure->Next();
            }

            base_node->AppendChild(std::move(node));
        }

    public:
        OgexParser() = default;
        virtual ~OgexParser() = default;

        virtual std::unique_ptr<Scene> Parse(const std::string& buf)
        {
            std::unique_ptr<Scene> pScene(new Scene("OGEX Scene"));
            OGEX::OpenGexDataDescription  openGexDataDescription;

            ODDL::DataResult result = openGexDataDescription.ProcessText(buf.c_str());
            if (result == ODDL::kDataOkay)
            {
                const ODDL::Structure* structure = openGexDataDescription.GetRootStructure()->GetFirstSubnode();
                while (structure)
                {
                    ConvertOddlStructureToSceneNode(*structure, pScene->SceneGraph, *pScene);

                    structure = structure->Next();
                }
            }

            return pScene;
        }
    private:
        bool m_bUpIsYAxis;
    };
}

