#include "OGEX.hpp"

using namespace My;

std::unique_ptr<Scene> OgexParser::Parse(const std::string& buf) {
    std::unique_ptr<Scene> pScene = make_unique<Scene>("OGEX Scene");
    OGEX::OpenGexDataDescription openGexDataDescription;

    ODDL::DataResult result = openGexDataDescription.ProcessText(buf.c_str());
    if (result == ODDL::kDataOkay) {
        const ODDL::Structure* structure =
            openGexDataDescription.GetRootStructure()->GetFirstSubnode();

        while (structure) {
            ConvertOddlStructureToSceneNode(*structure, pScene->SceneGraph,
                                            *pScene);

            structure = structure->Next();
        }
    }

    return pScene;
}

void OgexParser::ConvertOddlStructureToSceneNode(
    const ODDL::Structure& structure, std::shared_ptr<BaseSceneNode>& base_node,
    Scene& scene) {
    std::shared_ptr<BaseSceneNode> node;

    switch (structure.GetStructureType()) {
        case OGEX::kStructureMetric: {
            const auto& _structure =
                dynamic_cast<const OGEX::MetricStructure&>(structure);
            auto _key = _structure.GetMetricKey();
            const ODDL::Structure* sub_structure =
                _structure.GetFirstCoreSubnode();
            if (_key == "up") {
                const auto* dataStructure = static_cast<
                    const ODDL::DataStructure<ODDL::StringDataType>*>(
                    sub_structure);
                auto axis_name = dataStructure->GetDataElement(0);
                m_bUpIsYAxis = axis_name == "y";
            }
        }
            return;
        case OGEX::kStructureNode: {
            node =
                std::make_shared<SceneEmptyNode>(structure.GetStructureName());
        } break;
        case OGEX::kStructureBoneNode: {
            auto _node =
                std::make_shared<SceneBoneNode>(structure.GetStructureName());
            std::string _key = structure.GetStructureName();
            scene.BoneNodes.emplace(_key, _node);
            node = _node;
        } break;
        case OGEX::kStructureGeometryNode: {
            std::string _key = structure.GetStructureName();
            auto _node = std::make_shared<SceneGeometryNode>(_key);
            const auto& _structure =
                dynamic_cast<const OGEX::GeometryNodeStructure&>(structure);

            _node->SetVisibility(_structure.GetVisibleFlag());
            _node->SetIfCastShadow(_structure.GetShadowFlag());
            _node->SetIfMotionBlur(_structure.GetMotionBlurFlag());

            scene.GeometryNodes.emplace(_key, _node);

            // ref scene objects
            _key = _structure.GetObjectStructure()->GetStructureName();
            _node->AddSceneObjectRef(_key);

            // ref materials
            auto materials = _structure.GetMaterialStructureArray();
            auto materials_count = materials.GetElementCount();
            for (auto i = 0; i < materials_count; i++) {
                auto material = materials[i];
                _key = material->GetStructureName();
                _node->AddMaterialRef(_key);
            }

            std::string name = _structure.GetNodeName();
            scene.LUT_Name_GeometryNode.emplace(name, _node);

            node = _node;
        } break;
        case OGEX::kStructureLightNode: {
            auto _node =
                std::make_shared<SceneLightNode>(structure.GetStructureName());
            const auto& _structure =
                dynamic_cast<const OGEX::LightNodeStructure&>(structure);

            _node->SetIfCastShadow(_structure.GetShadowFlag());

            // ref scene objects
            std::string _key =
                _structure.GetObjectStructure()->GetStructureName();
            _node->AddSceneObjectRef(_key);

            scene.LightNodes.emplace(_key, _node);

            node = _node;
        } break;
        case OGEX::kStructureCameraNode: {
            auto _node =
                std::make_shared<SceneCameraNode>(structure.GetStructureName());
            const auto& _structure =
                dynamic_cast<const OGEX::CameraNodeStructure&>(structure);

            // ref scene objects
            std::string _key =
                _structure.GetObjectStructure()->GetStructureName();
            _node->AddSceneObjectRef(_key);

            scene.CameraNodes.emplace(_key, _node);

            node = _node;
        } break;
        case OGEX::kStructureGeometryObject: {
            const auto& _structure =
                dynamic_cast<const OGEX::GeometryObjectStructure&>(structure);
            std::string _key = _structure.GetStructureName();
            auto _object = std::make_shared<SceneObjectGeometry>();

            // properties
            _object->SetVisibility(_structure.GetVisibleFlag());
            _object->SetIfCastShadow(_structure.GetShadowFlag());
            _object->SetIfMotionBlur(_structure.GetMotionBlurFlag());

            // extensions
            //// collision shape
            ODDL::Structure* extension = _structure.GetFirstExtensionSubnode();
            while (extension) {
                const auto* _extension =
                    dynamic_cast<const OGEX::ExtensionStructure*>(extension);
                auto _appid = _extension->GetApplicationString();
                if (_appid == "MyGameEngine") {
                    auto _type = _extension->GetTypeString();
                    if (_type == "collision") {
                        const ODDL::Structure* sub_structure =
                            _extension->GetFirstCoreSubnode();
                        const auto* dataStructure1 = static_cast<
                            const ODDL::DataStructure<ODDL::StringDataType>*>(
                            sub_structure);
                        auto collision_type = dataStructure1->GetDataElement(0);

                        sub_structure = _extension->GetLastCoreSubnode();
                        const auto* dataStructure2 = static_cast<
                            const ODDL::DataStructure<ODDL::FloatDataType>*>(
                            sub_structure);
                        auto elementCount =
                            dataStructure2->GetDataElementCount();
                        auto* _data =
                            (float*)&dataStructure2->GetDataElement(0);
                        if (collision_type == "plane") {
                            _object->SetCollisionType(
                                SceneObjectCollisionType::
                                    kSceneObjectCollisionTypePlane);
                            _object->SetCollisionParameters(_data,
                                                            elementCount);
                        } else if (collision_type == "sphere") {
                            _object->SetCollisionType(
                                SceneObjectCollisionType::
                                    kSceneObjectCollisionTypeSphere);
                            _object->SetCollisionParameters(_data,
                                                            elementCount);
                        } else if (collision_type == "box") {
                            _object->SetCollisionType(
                                SceneObjectCollisionType::
                                    kSceneObjectCollisionTypeBox);
                            _object->SetCollisionParameters(_data,
                                                            elementCount);
                        }
                        break;
                    }
                }
                extension = extension->Next();
            }

            // meshs
            const ODDL::Map<OGEX::MeshStructure>* _meshs =
                _structure.GetMeshMap();
            int32_t _count = _meshs->GetElementCount();
            for (int32_t i = 0; i < _count; i++) {
                const OGEX::MeshStructure* _mesh = (*_meshs)[i];
                std::shared_ptr<SceneObjectMesh> mesh =
                    make_shared<SceneObjectMesh>();
                const std::string _primitive_type =
                    static_cast<const char*>(_mesh->GetMeshPrimitive());
                if (_primitive_type == "points") {
                    mesh->SetPrimitiveType(
                        PrimitiveType::kPrimitiveTypePointList);
                } else if (_primitive_type == "lines") {
                    mesh->SetPrimitiveType(
                        PrimitiveType::kPrimitiveTypeLineList);
                } else if (_primitive_type == "line_strip") {
                    mesh->SetPrimitiveType(
                        PrimitiveType::kPrimitiveTypeLineStrip);
                } else if (_primitive_type == "triangles") {
                    mesh->SetPrimitiveType(
                        PrimitiveType::kPrimitiveTypeTriList);
                } else if (_primitive_type == "triangle_strip") {
                    mesh->SetPrimitiveType(
                        PrimitiveType::kPrimitiveTypeTriStrip);
                } else if (_primitive_type == "quads") {
                    mesh->SetPrimitiveType(
                        PrimitiveType::kPrimitiveTypeQuadList);
                } else {
                    // not supported
                    mesh.reset();
                }
                if (mesh) {
                    const ODDL::Structure* sub_structure =
                        _mesh->GetFirstSubnode();
                    while (sub_structure) {
                        switch (sub_structure->GetStructureType()) {
                            case OGEX::kStructureVertexArray: {
                                const auto* _v = dynamic_cast<
                                    const OGEX::VertexArrayStructure*>(
                                    sub_structure);
                                const char* attr = _v->GetArrayAttrib();
                                auto morph_index = _v->GetMorphIndex();

                                const ODDL::Structure* _data_structure =
                                    _v->GetFirstCoreSubnode();
                                const auto* dataStructure = dynamic_cast<
                                    const ODDL::DataStructure<FloatDataType>*>(
                                    _data_structure);

                                auto arraySize = dataStructure->GetArraySize();
                                auto elementCount =
                                    dataStructure->GetDataElementCount();
                                const void* _data =
                                    &dataStructure->GetDataElement(0);
                                void* data = new float[elementCount];
                                size_t buf_size = sizeof(float) * elementCount;
                                memcpy(data, _data, buf_size);
                                VertexDataType vertexDataType;
                                switch (arraySize) {
                                    case 1:
                                        vertexDataType = VertexDataType::
                                            kVertexDataTypeFloat1;
                                        break;
                                    case 2:
                                        vertexDataType = VertexDataType::
                                            kVertexDataTypeFloat2;
                                        break;
                                    case 3:
                                        vertexDataType = VertexDataType::
                                            kVertexDataTypeFloat3;
                                        break;
                                    case 4:
                                        vertexDataType = VertexDataType::
                                            kVertexDataTypeFloat4;
                                        break;
                                    default:
                                        continue;
                                }
                                mesh->AddVertexArray(SceneObjectVertexArray(
                                    attr, morph_index, vertexDataType,
                                    (uint8_t*)data, elementCount));
                            } break;
                            case OGEX::kStructureIndexArray: {
                                const auto* _i = dynamic_cast<
                                    const OGEX::IndexArrayStructure*>(
                                    sub_structure);
                                auto material_index = _i->GetMaterialIndex();
                                auto restart_index = _i->GetRestartIndex();
                                const ODDL::Structure* _data_structure =
                                    _i->GetFirstCoreSubnode();
                                ODDL::StructureType type =
                                    _data_structure->GetStructureType();
                                int32_t elementCount = 0;
                                const void* _data = nullptr;
                                IndexDataType index_type =
                                    IndexDataType::kIndexDataTypeInt16;
                                switch (type) {
                                    case ODDL::kDataUnsignedInt8: {
                                        index_type =
                                            IndexDataType::kIndexDataTypeInt8;
                                        const auto* dataStructure =
                                            dynamic_cast<
                                                const ODDL::DataStructure<
                                                    UnsignedInt8DataType>*>(
                                                _data_structure);
                                        elementCount =
                                            dataStructure
                                                ->GetDataElementCount();
                                        _data =
                                            &dataStructure->GetDataElement(0);

                                    } break;
                                    case ODDL::kDataUnsignedInt16: {
                                        index_type =
                                            IndexDataType::kIndexDataTypeInt16;
                                        const auto* dataStructure =
                                            dynamic_cast<
                                                const ODDL::DataStructure<
                                                    UnsignedInt16DataType>*>(
                                                _data_structure);
                                        elementCount =
                                            dataStructure
                                                ->GetDataElementCount();
                                        _data =
                                            &dataStructure->GetDataElement(0);

                                    } break;
                                    case ODDL::kDataUnsignedInt32: {
                                        index_type =
                                            IndexDataType::kIndexDataTypeInt32;
                                        const auto* dataStructure =
                                            dynamic_cast<
                                                const ODDL::DataStructure<
                                                    UnsignedInt32DataType>*>(
                                                _data_structure);
                                        elementCount =
                                            dataStructure
                                                ->GetDataElementCount();
                                        _data =
                                            &dataStructure->GetDataElement(0);

                                    } break;
                                    case ODDL::kDataUnsignedInt64: {
                                        index_type =
                                            IndexDataType::kIndexDataTypeInt64;
                                        const auto* dataStructure =
                                            dynamic_cast<
                                                const ODDL::DataStructure<
                                                    UnsignedInt64DataType>*>(
                                                _data_structure);
                                        elementCount =
                                            dataStructure
                                                ->GetDataElementCount();
                                        _data =
                                            &dataStructure->GetDataElement(0);

                                    } break;
                                    default:;
                                }

                                int32_t data_size = 0;
                                switch (index_type) {
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
                                    default:;
                                }

                                size_t buf_size = elementCount * data_size;
                                void* data = new uint8_t[buf_size];
                                memcpy(data, _data, buf_size);
                                mesh->AddIndexArray(SceneObjectIndexArray(
                                    material_index, restart_index, index_type,
                                    (uint8_t*)data, elementCount));
                            } break;
                            default:
                                // ignore it
                                ;
                        }

                        sub_structure = sub_structure->Next();
                    }

                    _object->AddMesh(std::move(mesh));
                }
            }

            scene.Geometries[_key] = _object;
        }
            return;
        case OGEX::kStructureTransform: {
            int32_t index, count;
            const auto& _structure =
                dynamic_cast<const OGEX::TransformStructure&>(structure);
            bool object_flag = _structure.GetObjectFlag();
            Matrix4X4f matrix;
            std::shared_ptr<SceneObjectTransform> transform;

            auto _key = _structure.GetStructureName();
            count = _structure.GetTransformCount();
            for (index = 0; index < count; index++) {
                const float* data = _structure.GetTransform(index);
                matrix = data;
                if (!m_bUpIsYAxis) {
                    // commented out due to camera is in same coordinations
                    // so no need to exchange.
                    // exchange y and z
                    // ExchangeYandZ(matrix);
                }
                transform =
                    std::make_shared<SceneObjectTransform>(matrix, object_flag);
                base_node->AppendTransform(_key, transform);
            }
        }
            return;
        case OGEX::kStructureTranslation: {
            const auto& _structure =
                dynamic_cast<const OGEX::TranslationStructure&>(structure);
            bool object_flag = _structure.GetObjectFlag();
            std::shared_ptr<SceneObjectTranslation> translation;

            auto kind = _structure.GetTranslationKind();
            auto data = _structure.GetTranslation();
            if (kind == "xyz") {
                translation = std::make_shared<SceneObjectTranslation>(
                    data[0], data[1], data[2], object_flag);
            } else {
                translation = std::make_shared<SceneObjectTranslation>(
                    kind[0], data[0], object_flag);
            }
            auto _key = _structure.GetStructureName();
            base_node->AppendTransform(_key, std::move(translation));
        }
            return;
        case OGEX::kStructureRotation: {
            const auto& _structure =
                dynamic_cast<const OGEX::RotationStructure&>(structure);
            bool object_flag = _structure.GetObjectFlag();
            std::shared_ptr<SceneObjectRotation> rotation;

            auto kind = _structure.GetRotationKind();
            auto data = _structure.GetRotation();
            if (kind == "x") {
                rotation = std::make_shared<SceneObjectRotation>('x', data[0],
                                                                 object_flag);
            } else if (kind == "y") {
                rotation = std::make_shared<SceneObjectRotation>('y', data[0],
                                                                 object_flag);
            } else if (kind == "z") {
                rotation = std::make_shared<SceneObjectRotation>('z', data[0],
                                                                 object_flag);
            } else if (kind == "axis") {
                rotation = std::make_shared<SceneObjectRotation>(
                    Vector3f({data[0], data[1], data[2]}), data[3],
                    object_flag);
            } else if (kind == "quaternion") {
                rotation = std::make_shared<SceneObjectRotation>(
                    Quaternion<float>({data[0], data[1], data[2], data[3]}),
                    object_flag);
            }
            auto _key = _structure.GetStructureName();
            base_node->AppendTransform(_key, std::move(rotation));
        }
            return;
        case OGEX::kStructureScale: {
            const auto& _structure =
                dynamic_cast<const OGEX::ScaleStructure&>(structure);
            bool object_flag = _structure.GetObjectFlag();
            std::shared_ptr<SceneObjectScale> scale;

            auto kind = _structure.GetScaleKind();
            auto data = _structure.GetScale();
            if (kind == "x") {
                scale = std::make_shared<SceneObjectScale>('x', data[0],
                                                           object_flag);
            } else if (kind == "y") {
                scale = std::make_shared<SceneObjectScale>('y', data[0],
                                                           object_flag);
            } else if (kind == "z") {
                scale = std::make_shared<SceneObjectScale>('z', data[0],
                                                           object_flag);
            } else if (kind == "xyz") {
                scale = std::make_shared<SceneObjectScale>(
                    data[0], data[1], data[2], object_flag);
            }
            auto _key = _structure.GetStructureName();
            base_node->AppendTransform(_key, std::move(scale));
        }
            return;
        case OGEX::kStructureMaterial: {
            const auto& _structure =
                dynamic_cast<const OGEX::MaterialStructure&>(structure);
            std::string material_name;
            const char* _name = _structure.GetMaterialName();
            std::string _key = _structure.GetStructureName();
            auto material = std::make_shared<SceneObjectMaterial>();
            material->SetName(_name);

            const ODDL::Structure* _sub_structure =
                _structure.GetFirstCoreSubnode();
            while (_sub_structure) {
                std::string attrib, textureName;
                Vector4f color;
                float param;
                switch (_sub_structure->GetStructureType()) {
                    case OGEX::kStructureColor: {
                        attrib = dynamic_cast<const OGEX::ColorStructure*>(
                                     _sub_structure)
                                     ->GetAttribString();
                        color = dynamic_cast<const OGEX::ColorStructure*>(
                                    _sub_structure)
                                    ->GetColor();
                        material->SetColor(attrib, color);
                    } break;
                    case OGEX::kStructureParam: {
                        attrib = dynamic_cast<const OGEX::ParamStructure*>(
                                     _sub_structure)
                                     ->GetAttribString();
                        param = dynamic_cast<const OGEX::ParamStructure*>(
                                    _sub_structure)
                                    ->GetParam();
                        material->SetParam(attrib, param);
                    } break;
                    case OGEX::kStructureTexture: {
                        attrib = dynamic_cast<const OGEX::TextureStructure*>(
                                     _sub_structure)
                                     ->GetAttribString();
                        textureName =
                            dynamic_cast<const OGEX::TextureStructure*>(
                                _sub_structure)
                                ->GetTextureName();
                        material->SetTexture(attrib, textureName);
                    } break;
                    default:;
                };

                _sub_structure = _sub_structure->Next();
            }
            scene.Materials[_key] = material;
        }
            return;
        case OGEX::kStructureLightObject: {
            const auto& _structure =
                dynamic_cast<const OGEX::LightObjectStructure&>(structure);
            const char* _type_str = _structure.GetTypeString();
            const bool _bshadow = _structure.GetShadowFlag();
            std::string _key = _structure.GetStructureName();
            std::shared_ptr<SceneObjectLight> light;

            if (!strncmp(_type_str, "infinite", 8)) {
                light = std::make_shared<SceneObjectInfiniteLight>();
            } else if (!strncmp(_type_str, "point", 5)) {
                light = std::make_shared<SceneObjectOmniLight>();
            } else if (!strncmp(_type_str, "spot", 4)) {
                light = std::make_shared<SceneObjectSpotLight>();
            } else if (!strncmp(_type_str, "area", 4)) {
                light = std::make_shared<SceneObjectAreaLight>();
            } else {
                assert(0);
            }

            light->SetIfCastShadow(_bshadow);

            const ODDL::Structure* _sub_structure =
                _structure.GetFirstCoreSubnode();
            while (_sub_structure) {
                std::string attrib, textureName;
                Vector4f color;
                float param;
                switch (_sub_structure->GetStructureType()) {
                    case OGEX::kStructureColor: {
                        attrib = dynamic_cast<const OGEX::ColorStructure*>(
                                     _sub_structure)
                                     ->GetAttribString();
                        color = dynamic_cast<const OGEX::ColorStructure*>(
                                    _sub_structure)
                                    ->GetColor();
                        light->SetColor(attrib, color);
                    } break;
                    case OGEX::kStructureParam: {
                        attrib = dynamic_cast<const OGEX::ParamStructure*>(
                                     _sub_structure)
                                     ->GetAttribString();
                        param = dynamic_cast<const OGEX::ParamStructure*>(
                                    _sub_structure)
                                    ->GetParam();
                        light->SetParam(attrib, param);
                    } break;
                    case OGEX::kStructureTexture: {
                        attrib = dynamic_cast<const OGEX::TextureStructure*>(
                                     _sub_structure)
                                     ->GetAttribString();
                        textureName =
                            dynamic_cast<const OGEX::TextureStructure*>(
                                _sub_structure)
                                ->GetTextureName();
                        light->SetTexture(attrib, textureName);
                    } break;
                    case OGEX::kStructureAtten: {
                        auto atten = dynamic_cast<const OGEX::AttenStructure*>(
                            _sub_structure);
                        AttenCurve curve;
                        if (atten->GetCurveType() == "linear") {
                            curve.type = AttenCurveType::kLinear;
                            curve.u.linear_params.begin_atten =
                                atten->GetBeginParam();
                            curve.u.linear_params.end_atten =
                                atten->GetEndParam();
                        } else if (atten->GetCurveType() == "smooth") {
                            curve.type = AttenCurveType::kSmooth;
                            curve.u.smooth_params.begin_atten =
                                atten->GetBeginParam();
                            curve.u.smooth_params.end_atten =
                                atten->GetEndParam();
                        } else if (atten->GetCurveType() == "inverse") {
                            curve.type = AttenCurveType::kInverse;
                            curve.u.inverse_params.scale =
                                atten->GetScaleParam();
                            curve.u.inverse_params.offset =
                                atten->GetOffsetParam();
                            curve.u.inverse_params.kl = atten->GetLinearParam();
                            curve.u.inverse_params.kc =
                                atten->GetConstantParam();
                        } else if (atten->GetCurveType() == "inverse_square") {
                            curve.type = AttenCurveType::kInverseSquare;
                            curve.u.inverse_squre_params.scale =
                                atten->GetScaleParam();
                            curve.u.inverse_squre_params.offset =
                                atten->GetOffsetParam();
                            curve.u.inverse_squre_params.kq =
                                atten->GetQuadraticParam();
                            curve.u.inverse_squre_params.kl =
                                atten->GetLinearParam();
                            curve.u.inverse_squre_params.kc =
                                atten->GetConstantParam();
                        }

                        if (atten->GetAttenKind() == "angle") {
                            auto _light =
                                dynamic_pointer_cast<SceneObjectSpotLight>(
                                    light);
                            _light->SetAngleAttenuation(curve);
                        } else if (atten->GetAttenKind() == "cos_angle") {
                            // TODO: mark the angle is in cos value instead
                            // of rad
                            auto _light =
                                dynamic_pointer_cast<SceneObjectSpotLight>(
                                    light);
                            _light->SetAngleAttenuation(curve);
                        } else {
                            light->SetDistanceAttenuation(curve);
                        }
                    } break;
                    default:;
                };

                _sub_structure = _sub_structure->Next();
            }

            // extensions
            ODDL::Structure* extension = _structure.GetFirstExtensionSubnode();
            while (extension) {
                const auto* _extension =
                    dynamic_cast<const OGEX::ExtensionStructure*>(extension);
                auto _appid = _extension->GetApplicationString();
                if (_appid == "MyGameEngine") {
                    auto _type = _extension->GetTypeString();
                    if (_type == "area_light") {
                        const ODDL::Structure* sub_structure =
                            _extension->GetFirstCoreSubnode();
                        const auto* dataStructure1 = static_cast<
                            const ODDL::DataStructure<ODDL::FloatDataType>*>(
                            sub_structure);
                        auto elementCount =
                            dataStructure1->GetDataElementCount();
                        assert(elementCount == 2);
                        auto width = dataStructure1->GetDataElement(0);
                        auto height = dataStructure1->GetDataElement(1);

                        auto _light =
                            dynamic_pointer_cast<SceneObjectAreaLight>(light);
                        _light->SetDimension({width, height});
                    }
                }
                extension = extension->Next();
            }

            scene.Lights[_key] = light;
        }
            return;
        case OGEX::kStructureCameraObject: {
            const auto& _structure =
                dynamic_cast<const OGEX::CameraObjectStructure&>(structure);
            std::string _key = _structure.GetStructureName();
            auto camera = std::make_shared<SceneObjectPerspectiveCamera>();

            const ODDL::Structure* _sub_structure =
                _structure.GetFirstCoreSubnode();
            while (_sub_structure) {
                std::string attrib, textureName;
                Vector4f color;
                float param;
                switch (_sub_structure->GetStructureType()) {
                    case OGEX::kStructureColor: {
                        attrib = dynamic_cast<const OGEX::ColorStructure*>(
                                     _sub_structure)
                                     ->GetAttribString();
                        color = dynamic_cast<const OGEX::ColorStructure*>(
                                    _sub_structure)
                                    ->GetColor();
                        camera->SetColor(attrib, color);
                    } break;
                    case OGEX::kStructureParam: {
                        attrib = dynamic_cast<const OGEX::ParamStructure*>(
                                     _sub_structure)
                                     ->GetAttribString();
                        param = dynamic_cast<const OGEX::ParamStructure*>(
                                    _sub_structure)
                                    ->GetParam();
                        camera->SetParam(attrib, param);
                    } break;
                    case OGEX::kStructureTexture: {
                        attrib = dynamic_cast<const OGEX::TextureStructure*>(
                                     _sub_structure)
                                     ->GetAttribString();
                        textureName =
                            dynamic_cast<const OGEX::TextureStructure*>(
                                _sub_structure)
                                ->GetTextureName();
                        camera->SetTexture(attrib, textureName);
                    } break;
                    default:;
                };

                _sub_structure = _sub_structure->Next();
            }
            scene.Cameras[_key] = camera;
        }
            return;
        case OGEX::kStructureAnimation: {
            const auto& _structure =
                dynamic_cast<const OGEX::AnimationStructure&>(structure);
            auto clip_index = _structure.GetClipIndex();
            std::shared_ptr<SceneObjectAnimationClip> clip =
                std::make_shared<SceneObjectAnimationClip>(clip_index);

            const ODDL::Structure* _sub_structure =
                _structure.GetFirstCoreSubnode();
            while (_sub_structure) {
                switch (_sub_structure->GetStructureType()) {
                    case OGEX::kStructureTrack: {
                        const auto& track_structure =
                            dynamic_cast<const OGEX::TrackStructure&>(
                                *_sub_structure);
                        const auto& time_structure =
                            dynamic_cast<const OGEX::TimeStructure&>(
                                *track_structure.GetTimeStructure());
                        const auto& value_structure =
                            dynamic_cast<const OGEX::ValueStructure&>(
                                *track_structure.GetValueStructure());
                        auto ref = track_structure.GetTargetRef();
                        std::string _key(*ref.GetNameArray());
                        std::shared_ptr<SceneObjectTransform> trans;
                        trans = base_node->GetTransform(_key);
                        std::shared_ptr<SceneObjectTrack> track;

                        auto time_key_value =
                            time_structure.GetKeyValueStructure();
                        auto time_key_data_count =
                            time_structure.GetKeyDataElementCount();
                        auto dataStructure = static_cast<
                            const ODDL::DataStructure<ODDL::FloatDataType>*>(
                            time_key_value->GetFirstCoreSubnode());
                        auto time_array_size = dataStructure->GetArraySize();
                        const float* time_knots =
                            &dataStructure->GetDataElement(0);
                        // current we only handle 1D time curve
                        assert(time_array_size == 0);

                        auto value_key_value =
                            value_structure.GetKeyValueStructure();
                        auto value_key_data_count =
                            value_structure.GetKeyDataElementCount();
                        dataStructure = static_cast<
                            const ODDL::DataStructure<ODDL::FloatDataType>*>(
                            value_key_value->GetFirstCoreSubnode());
                        auto value_array_size = dataStructure->GetArraySize();
                        const float* value_knots =
                            &dataStructure->GetDataElement(0);
                        std::shared_ptr<CurveBase> time_curve;
                        std::shared_ptr<CurveBase> value_curve;
                        SceneObjectTrackType type =
                            SceneObjectTrackType::kScalar;

                        if (time_structure.GetCurveType() == "bezier") {
                            auto key_incoming_control =
                                time_structure.GetKeyControlStructure(0);
                            auto key_outgoing_control =
                                time_structure.GetKeyControlStructure(1);
                            dataStructure =
                                static_cast<const ODDL::DataStructure<
                                    ODDL::FloatDataType>*>(
                                    key_incoming_control
                                        ->GetFirstCoreSubnode());
                            const float* in_cp =
                                &dataStructure->GetDataElement(0);
                            dataStructure =
                                static_cast<const ODDL::DataStructure<
                                    ODDL::FloatDataType>*>(
                                    key_outgoing_control
                                        ->GetFirstCoreSubnode());
                            const float* out_cp =
                                &dataStructure->GetDataElement(0);
                            time_curve = std::make_shared<Bezier<float, float>>(
                                time_knots, in_cp, out_cp, time_key_data_count);
                        } else {
                            time_curve = std::make_shared<Linear<float, float>>(
                                time_knots, time_key_data_count);
                        }

                        if (value_structure.GetCurveType() == "bezier") {
                            auto key_incoming_control =
                                value_structure.GetKeyControlStructure(0);
                            auto key_outgoing_control =
                                value_structure.GetKeyControlStructure(1);
                            dataStructure =
                                static_cast<const ODDL::DataStructure<
                                    ODDL::FloatDataType>*>(
                                    key_incoming_control
                                        ->GetFirstCoreSubnode());
                            const float* in_cp =
                                &dataStructure->GetDataElement(0);
                            dataStructure =
                                static_cast<const ODDL::DataStructure<
                                    ODDL::FloatDataType>*>(
                                    key_outgoing_control
                                        ->GetFirstCoreSubnode());
                            const float* out_cp =
                                &dataStructure->GetDataElement(0);

                            switch (value_array_size) {
                                case 0:
                                case 1: {
                                    value_curve =
                                        std::make_shared<Bezier<float, float>>(
                                            value_knots, in_cp, out_cp,
                                            value_key_data_count);
                                    type = SceneObjectTrackType::kScalar;
                                } break;
                                case 3: {
                                    value_curve = std::make_shared<
                                        Bezier<Vector3f, Vector3f>>(
                                        reinterpret_cast<const Vector3f*>(
                                            value_knots),
                                        reinterpret_cast<const Vector3f*>(
                                            in_cp),
                                        reinterpret_cast<const Vector3f*>(
                                            out_cp),
                                        value_key_data_count);
                                    type = SceneObjectTrackType::kVector3;
                                } break;
                                case 4: {
                                    value_curve = std::make_shared<
                                        Bezier<Quaternion<float>, float>>(
                                        reinterpret_cast<
                                            const Quaternion<float>*>(
                                            value_knots),
                                        reinterpret_cast<
                                            const Quaternion<float>*>(in_cp),
                                        reinterpret_cast<
                                            const Quaternion<float>*>(out_cp),
                                        value_key_data_count);
                                    type = SceneObjectTrackType::kQuoternion;
                                } break;
                                case 16: {
                                    value_curve = std::make_shared<
                                        Bezier<Matrix4X4f, float>>(
                                        reinterpret_cast<const Matrix4X4f*>(
                                            value_knots),
                                        reinterpret_cast<const Matrix4X4f*>(
                                            in_cp),
                                        reinterpret_cast<const Matrix4X4f*>(
                                            out_cp),
                                        value_key_data_count);
                                    type = SceneObjectTrackType::kMatrix;
                                } break;
                                default:
                                    assert(0);
                            }
                        } else  // default to linear
                        {
                            switch (value_array_size) {
                                case 0:
                                case 1: {
                                    value_curve =
                                        std::make_shared<Linear<float, float>>(
                                            value_knots, value_key_data_count);
                                    type = SceneObjectTrackType::kScalar;
                                } break;
                                case 3: {
                                    value_curve = std::make_shared<
                                        Linear<Vector3f, Vector3f>>(
                                        reinterpret_cast<const Vector3f*>(
                                            value_knots),
                                        value_key_data_count);
                                    type = SceneObjectTrackType::kVector3;
                                } break;
                                case 4: {
                                    value_curve = std::make_shared<
                                        Linear<Quaternion<float>, float>>(
                                        reinterpret_cast<
                                            const Quaternion<float>*>(
                                            value_knots),
                                        value_key_data_count);
                                    type = SceneObjectTrackType::kQuoternion;
                                } break;
                                case 16: {
                                    value_curve = std::make_shared<
                                        Linear<Matrix4X4f, float>>(
                                        reinterpret_cast<const Matrix4X4f*>(
                                            value_knots),
                                        value_key_data_count);
                                    type = SceneObjectTrackType::kMatrix;
                                } break;
                                default:
                                    assert(0);
                            }
                        }

                        track = std::make_shared<SceneObjectTrack>(
                            trans, time_curve, value_curve, type);

                        clip->AddTrack(track);
                    } break;
                    default:;
                };

                _sub_structure = _sub_structure->Next();
            }

            base_node->AttachAnimationClip(clip_index, clip);

            // register the node to animatable node LUT
            scene.AnimatableNodes.push_back(base_node);
        }

            return;
        default:
            // just ignore it and finish
            return;
    };

    const ODDL::Structure* sub_structure = structure.GetFirstSubnode();
    while (sub_structure) {
        ConvertOddlStructureToSceneNode(*sub_structure, node, scene);

        sub_structure = sub_structure->Next();
    }

    base_node->AppendChild(std::move(node));
}
