#include "OpenGEX.h"

#include "OGEX.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "Bezier.hpp"
#include "Curve.hpp"
#include "Linear.hpp"

using namespace My;

namespace {
// ODDL parses nested structures recursively; bound the depth so that hostile
// input cannot exhaust the stack.
constexpr int kMaxNestingDepth = 128;

void Report(const std::string& message) {
    std::cerr << "[OgexParser] " << message << std::endl;
}

std::string ToStr(const ODDL::String& text) {
    return std::string(static_cast<const char*>(text));
}

template <class T>
const T* Cast(const ODDL::Structure* structure) {
    return dynamic_cast<const T*>(structure);
}

bool AllFinite(const float* values, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (!std::isfinite(values[i])) return false;
    }
    return true;
}

// Rejects embedded NULs and excessive brace nesting before ODDL sees the text.
bool IsAcceptableText(const std::string& text) {
    if (text.find('\0') != std::string::npos) {
        Report("OpenGEX text contains an embedded NUL");
        return false;
    }

    int depth = 0;
    const size_t size = text.size();
    for (size_t i = 0; i < size; i++) {
        const char c = text[i];
        if (c == '"' || c == '\'') {
            for (i++; i < size && text[i] != c; i++) {
                if (text[i] == '\\') i++;
            }
        } else if (c == '/' && i + 1 < size && text[i + 1] == '/') {
            while (i < size && text[i] != '\n') i++;
        } else if (c == '/' && i + 1 < size && text[i + 1] == '*') {
            const size_t end = text.find("*/", i + 2);
            if (end == std::string::npos) break;
            i = end + 1;
        } else if (c == '{') {
            if (++depth > kMaxNestingDepth) {
                Report("OpenGEX nesting is too deep");
                return false;
            }
        } else if (c == '}' && depth > 0) {
            depth--;
        }
    }
    return true;
}

struct ExtensionPayload {
    const char* name{nullptr};
    const float* values{nullptr};
    int32_t count{0};
};

// Extension bodies are an optional leading string followed by a float array.
bool ReadExtensionPayload(const OGEX::ExtensionStructure& extension,
                          bool expectString, ExtensionPayload& payload) {
    const auto* first = extension.GetFirstCoreSubnode();
    const auto* last = extension.GetLastCoreSubnode();
    if (!first) return false;

    const ODDL::Structure* floats = first;
    if (expectString) {
        const auto* text = dynamic_cast<
            const ODDL::DataStructure<ODDL::StringDataType>*>(first);
        if (!text || text->GetDataElementCount() != 1) return false;
        payload.name = text->GetDataElement(0);
        floats = (first == last) ? nullptr : last;
    } else if (first != last) {
        return false;
    }

    if (floats) {
        const auto* data =
            dynamic_cast<const ODDL::DataStructure<ODDL::FloatDataType>*>(
                floats);
        if (!data || data->GetDataElementCount() <= 0) return false;
        payload.values = &data->GetDataElement(0);
        payload.count = data->GetDataElementCount();
        if (!AllFinite(payload.values, payload.count)) return false;
    }
    return true;
}

bool ParseGeometryExtension(const OGEX::ExtensionStructure& extension,
                            SceneObjectGeometry& geometry,
                            SceneObjectPhysicsProperties& physics) {
    const std::string type = ToStr(extension.GetTypeString());
    ExtensionPayload payload;

    if (type == "collision") {
        if (!ReadExtensionPayload(extension, true, payload) || !payload.name) {
            return false;
        }
        const std::string_view shape(payload.name);
        struct Shape {
            std::string_view name;
            SceneObjectCollisionType type;
            int32_t parameters;
        };
        static constexpr Shape shapes[] = {
            {"plane", SceneObjectCollisionType::kSceneObjectCollisionTypePlane,
             4},
            {"sphere",
             SceneObjectCollisionType::kSceneObjectCollisionTypeSphere, 1},
            {"box", SceneObjectCollisionType::kSceneObjectCollisionTypeBox, 3},
            {"bvhmesh",
             SceneObjectCollisionType::kSceneObjectCollisionTypeBvhMesh, 0}};
        constexpr int32_t kMaxCollisionParameters = 10;
        for (const auto& candidate : shapes) {
            if (shape != candidate.name) continue;
            if (payload.count < candidate.parameters ||
                payload.count >= kMaxCollisionParameters) {
                return false;
            }
            geometry.SetCollisionType(candidate.type);
            if (payload.count > 0) {
                geometry.SetCollisionParameters(payload.values, payload.count);
            }
            return true;
        }
        return false;
    }

    if (type == "physics_material") {
        if (!ReadExtensionPayload(extension, false, payload) ||
            payload.count != 2 || payload.values[0] < 0.0f ||
            payload.values[1] < 0.0f) {
            return false;
        }
        physics.friction = payload.values[0];
        physics.restitution = payload.values[1];
        return true;
    }

    if (type == "rigid_body") {
        if (!ReadExtensionPayload(extension, true, payload) || !payload.name ||
            payload.count != 4) {
            return false;
        }
        const std::string_view body(payload.name);
        if (body == "dynamic") {
            physics.bodyType = SceneObjectBodyType::kDynamic;
        } else if (body == "static") {
            physics.bodyType = SceneObjectBodyType::kStatic;
        } else {
            return false;
        }
        if (payload.values[0] <= 0.0f) return false;
        physics.mass = payload.values[0];
        physics.linearVelocity =
            Vector3f{payload.values[1], payload.values[2], payload.values[3]};
        return true;
    }

    if (type == "motion_constraint") {
        if (!ReadExtensionPayload(extension, true, payload) || !payload.name ||
            std::string_view(payload.name) != "plane" || payload.count != 3) {
            return false;
        }
        int axis = -1;
        for (int i = 0; i < 3; i++) {
            const float v = std::fabs(payload.values[i]);
            if (v == 0.0f) continue;
            if (v != 1.0f || axis >= 0) return false;
            axis = i;
        }
        if (axis < 0) return false;
        physics.planeNormalAxis = axis;
        return true;
    }

    return true;
}

// Color / Param / Texture children are shared by materials, lights and cameras.
template <class Target>
bool ParseAttributeChildren(const ODDL::Structure* child, Target& target) {
    for (; child; child = child->Next()) {
        switch (child->GetStructureType()) {
            case OGEX::kStructureColor: {
                const auto* color = Cast<OGEX::ColorStructure>(child);
                if (!color) return false;
                std::string attrib = ToStr(color->GetAttribString());
                Vector4f value;
                value = color->GetColor();
                target.SetColor(attrib, value);
            } break;
            case OGEX::kStructureParam: {
                const auto* param = Cast<OGEX::ParamStructure>(child);
                if (!param) return false;
                std::string attrib = ToStr(param->GetAttribString());
                target.SetParam(attrib, param->GetParam());
            } break;
            case OGEX::kStructureTexture: {
                const auto* texture = Cast<OGEX::TextureStructure>(child);
                if (!texture) return false;
                std::string attrib = ToStr(texture->GetAttribString());
                std::string name = ToStr(texture->GetTextureName());
                target.SetTexture(attrib, name);
            } break;
            default:;
        }
    }
    return true;
}

bool ParseAtten(const OGEX::AttenStructure& atten, SceneObjectLight& light) {
    AttenCurve curve;
    const std::string_view curveType(
        static_cast<const char*>(atten.GetCurveType()));
    if (curveType == "linear") {
        curve.type = AttenCurveType::kLinear;
        curve.u.linear_params.begin_atten = atten.GetBeginParam();
        curve.u.linear_params.end_atten = atten.GetEndParam();
    } else if (curveType == "smooth") {
        curve.type = AttenCurveType::kSmooth;
        curve.u.smooth_params.begin_atten = atten.GetBeginParam();
        curve.u.smooth_params.end_atten = atten.GetEndParam();
    } else if (curveType == "inverse") {
        curve.type = AttenCurveType::kInverse;
        curve.u.inverse_params.scale = atten.GetScaleParam();
        curve.u.inverse_params.offset = atten.GetOffsetParam();
        curve.u.inverse_params.kl = atten.GetLinearParam();
        curve.u.inverse_params.kc = atten.GetConstantParam();
    } else if (curveType == "inverse_square") {
        curve.type = AttenCurveType::kInverseSquare;
        curve.u.inverse_squre_params.scale = atten.GetScaleParam();
        curve.u.inverse_squre_params.offset = atten.GetOffsetParam();
        curve.u.inverse_squre_params.kq = atten.GetQuadraticParam();
        curve.u.inverse_squre_params.kl = atten.GetLinearParam();
        curve.u.inverse_squre_params.kc = atten.GetConstantParam();
    } else {
        return false;
    }

    const std::string_view kind(static_cast<const char*>(atten.GetAttenKind()));
    if (kind == "angle" || kind == "cos_angle") {
        // TODO: mark whether the angle is in cos value instead of rad
        auto* spot = dynamic_cast<SceneObjectSpotLight*>(&light);
        if (!spot) return false;
        spot->SetAngleAttenuation(curve);
    } else {
        light.SetDistanceAttenuation(curve);
    }
    return true;
}

bool ParseLightExtension(const OGEX::ExtensionStructure& extension,
                         SceneObjectLight& light) {
    if (extension.GetApplicationString() != "MyGameEngine" ||
        extension.GetTypeString() != "area_light") {
        return true;
    }
    auto* area = dynamic_cast<SceneObjectAreaLight*>(&light);
    const auto* data = Cast<ODDL::DataStructure<ODDL::FloatDataType>>(
        extension.GetFirstCoreSubnode());
    if (!area || !data || data->GetDataElementCount() != 2 ||
        !AllFinite(&data->GetDataElement(0), 2)) {
        return false;
    }
    area->SetDimension({data->GetDataElement(0), data->GetDataElement(1)});
    return true;
}

struct IndexSource {
    IndexDataType type{IndexDataType::kIndexDataTypeInt16};
    const void* data{nullptr};
    size_t count{0};
    size_t elementSize{0};
};

template <class DataType>
bool TryIndexSource(const ODDL::Structure* structure, IndexDataType type,
                    IndexSource& source) {
    const auto* data =
        dynamic_cast<const ODDL::DataStructure<DataType>*>(structure);
    if (!data) return false;
    const int32_t count = data->GetDataElementCount();
    if (count <= 0) return false;
    source.type = type;
    source.data = &data->GetDataElement(0);
    source.count = static_cast<size_t>(count);
    source.elementSize = sizeof(typename DataType::PrimType);
    return true;
}

bool ReadIndexSource(const ODDL::Structure* structure, IndexSource& source) {
    if (!structure) return false;
    return TryIndexSource<ODDL::UnsignedInt8DataType>(
               structure, IndexDataType::kIndexDataTypeInt8, source) ||
           TryIndexSource<ODDL::UnsignedInt16DataType>(
               structure, IndexDataType::kIndexDataTypeInt16, source) ||
           TryIndexSource<ODDL::UnsignedInt32DataType>(
               structure, IndexDataType::kIndexDataTypeInt32, source) ||
           TryIndexSource<ODDL::UnsignedInt64DataType>(
               structure, IndexDataType::kIndexDataTypeInt64, source);
}

template <class T>
bool IndicesInRange(const void* data, size_t count, uint64_t vertexCount,
                    uint64_t restartIndex) {
    const T* indices = static_cast<const T*>(data);
    for (size_t i = 0; i < count; i++) {
        const uint64_t index = indices[i];
        if (index != restartIndex && index >= vertexCount) return false;
    }
    return true;
}

bool IndicesInRange(const IndexSource& source, uint64_t vertexCount,
                    uint64_t restartIndex) {
    switch (source.type) {
        case IndexDataType::kIndexDataTypeInt8:
            return IndicesInRange<uint8_t>(source.data, source.count,
                                           vertexCount, restartIndex);
        case IndexDataType::kIndexDataTypeInt16:
            return IndicesInRange<uint16_t>(source.data, source.count,
                                            vertexCount, restartIndex);
        case IndexDataType::kIndexDataTypeInt32:
            return IndicesInRange<uint32_t>(source.data, source.count,
                                            vertexCount, restartIndex);
        case IndexDataType::kIndexDataTypeInt64:
            return IndicesInRange<uint64_t>(source.data, source.count,
                                            vertexCount, restartIndex);
        default:
            return false;
    }
}

struct PrimitiveInfo {
    std::string_view name;
    PrimitiveType type;
    size_t indicesPerPrimitive;
};

constexpr PrimitiveInfo kPrimitives[] = {
    {"points", PrimitiveType::kPrimitiveTypePointList, 1},
    {"lines", PrimitiveType::kPrimitiveTypeLineList, 2},
    {"line_strip", PrimitiveType::kPrimitiveTypeLineStrip, 1},
    {"triangles", PrimitiveType::kPrimitiveTypeTriList, 3},
    {"triangle_strip", PrimitiveType::kPrimitiveTypeTriStrip, 1},
    {"quads", PrimitiveType::kPrimitiveTypeQuadList, 4}};

// Returns false on invalid data. Meshes with an unsupported primitive type are
// skipped (mesh left null) as before.
bool ParseMesh(const OGEX::MeshStructure& meshStructure,
               std::shared_ptr<SceneObjectMesh>& mesh) {
    const std::string_view primitiveName(
        static_cast<const char*>(meshStructure.GetMeshPrimitive()));
    const PrimitiveInfo* primitive = nullptr;
    for (const auto& candidate : kPrimitives) {
        if (candidate.name == primitiveName) primitive = &candidate;
    }
    if (!primitive) return true;

    mesh = std::make_shared<SceneObjectMesh>();
    mesh->SetPrimitiveType(primitive->type);

    // vertex arrays first, so index ranges can be checked whatever the order
    bool hasVertexCount = false;
    uint64_t vertexCount = 0;
    for (const ODDL::Structure* sub = meshStructure.GetFirstSubnode(); sub;
         sub = sub->Next()) {
        if (sub->GetStructureType() != OGEX::kStructureVertexArray) continue;
        const auto* vertexArray = Cast<OGEX::VertexArrayStructure>(sub);
        const auto* data =
            vertexArray ? Cast<ODDL::DataStructure<ODDL::FloatDataType>>(
                              vertexArray->GetFirstCoreSubnode())
                        : nullptr;
        if (!data || data->GetDataElementCount() <= 0) return false;

        const uint32_t arraySize = data->GetArraySize();
        const size_t elementCount =
            static_cast<size_t>(data->GetDataElementCount());
        VertexDataType vertexDataType;
        switch (arraySize) {
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
                continue;  // unsupported layout, ignored
        }

        const float* values = &data->GetDataElement(0);
        if (elementCount % arraySize != 0 || !AllFinite(values, elementCount)) {
            return false;
        }
        const uint64_t count = elementCount / arraySize;
        if (hasVertexCount && count != vertexCount) return false;
        hasVertexCount = true;
        vertexCount = count;

        auto buffer = std::make_unique<uint8_t[]>(sizeof(float) * elementCount);
        std::memcpy(buffer.get(), values, sizeof(float) * elementCount);
        mesh->AddVertexArray(SceneObjectVertexArray(
            vertexArray->GetArrayAttrib(), vertexArray->GetMorphIndex(),
            vertexDataType, buffer.release(), elementCount));
    }

    for (const ODDL::Structure* sub = meshStructure.GetFirstSubnode(); sub;
         sub = sub->Next()) {
        if (sub->GetStructureType() != OGEX::kStructureIndexArray) continue;
        const auto* indexArray = Cast<OGEX::IndexArrayStructure>(sub);
        IndexSource source;
        if (!indexArray ||
            !ReadIndexSource(indexArray->GetFirstCoreSubnode(), source)) {
            return false;
        }
        if (source.count % primitive->indicesPerPrimitive != 0) return false;
        const uint64_t restart = indexArray->GetRestartIndex();
        if (!hasVertexCount || !IndicesInRange(source, vertexCount, restart)) {
            return false;
        }

        const size_t bytes = source.count * source.elementSize;
        auto buffer = std::make_unique<uint8_t[]>(bytes);
        std::memcpy(buffer.get(), source.data, bytes);
        mesh->AddIndexArray(SceneObjectIndexArray(
            indexArray->GetMaterialIndex(), static_cast<size_t>(restart),
            source.type, buffer.release(), source.count));
    }
    return true;
}

struct KeyData {
    const float* values{nullptr};
    size_t floatCount{0};
    uint32_t arraySize{0};
};

bool ReadKeyData(const OGEX::KeyStructure* key, KeyData& out) {
    if (!key) return false;
    const auto* data = Cast<ODDL::DataStructure<ODDL::FloatDataType>>(
        key->GetFirstCoreSubnode());
    if (!data || data->GetDataElementCount() <= 0) return false;
    out.values = &data->GetDataElement(0);
    out.floatCount = static_cast<size_t>(data->GetDataElementCount());
    out.arraySize = data->GetArraySize();
    return AllFinite(out.values, out.floatCount);
}

// Copies packed floats into a typed vector without aliasing the source buffer.
template <class T>
std::vector<T> CopyKeys(const float* values, size_t keyCount) {
    static_assert(std::is_standard_layout_v<T>);
    std::vector<T> keys(keyCount);
    std::memcpy(static_cast<void*>(keys.data()), values, keyCount * sizeof(T));
    return keys;
}

template <class T, class P>
std::shared_ptr<CurveBase> MakeCurve(const KeyData& knots, const KeyData* in,
                                     const KeyData* out) {
    static_assert(sizeof(T) % sizeof(float) == 0);
    const size_t keyCount = knots.floatCount / (sizeof(T) / sizeof(float));
    if (in && out) {
        return std::make_shared<Bezier<T, P>>(
            CopyKeys<T>(knots.values, keyCount),
            CopyKeys<T>(in->values, keyCount),
            CopyKeys<T>(out->values, keyCount));
    }
    return std::make_shared<Linear<T, P>>(CopyKeys<T>(knots.values, keyCount));
}

bool ParseTrack(const OGEX::TrackStructure& track, BaseSceneNode& node,
                SceneObjectAnimationClip& clip) {
    const auto* time = Cast<OGEX::TimeStructure>(track.GetTimeStructure());
    const auto* value = Cast<OGEX::ValueStructure>(track.GetValueStructure());
    if (!time || !value) return false;

    const auto& names = track.GetTargetRef().GetNameArray();
    if (names.GetElementCount() < 1) return false;
    const std::string targetName = ToStr(names[0]);
    std::shared_ptr<SceneObjectTransform> transform =
        node.GetTransform(targetName);
    if (!transform) return false;

    KeyData timeKeys;
    if (!ReadKeyData(time->GetKeyValueStructure(), timeKeys) ||
        timeKeys.arraySize != 0) {
        return false;
    }
    const size_t keyCount = timeKeys.floatCount;
    for (size_t i = 1; i < keyCount; i++) {
        if (timeKeys.values[i] < timeKeys.values[i - 1]) return false;
    }

    KeyData valueKeys;
    if (!ReadKeyData(value->GetKeyValueStructure(), valueKeys)) return false;
    const size_t components = valueKeys.arraySize == 0 ? 1 : valueKeys.arraySize;
    if (valueKeys.floatCount != keyCount * components) return false;

    // control points must match their knots exactly
    auto readControls = [&](const OGEX::CurveStructure& curve, size_t floats,
                            KeyData& in, KeyData& out) {
        return ReadKeyData(curve.GetKeyControlStructure(0), in) &&
               ReadKeyData(curve.GetKeyControlStructure(1), out) &&
               in.floatCount == floats && out.floatCount == floats;
    };

    KeyData timeIn, timeOut, valueIn, valueOut;
    const bool timeBezier = time->GetCurveType() == "bezier";
    const bool valueBezier = value->GetCurveType() == "bezier";
    if (timeBezier && !readControls(*time, keyCount, timeIn, timeOut)) {
        return false;
    }
    if (valueBezier &&
        !readControls(*value, valueKeys.floatCount, valueIn, valueOut)) {
        return false;
    }

    std::shared_ptr<CurveBase> timeCurve =
        timeBezier ? MakeCurve<float, float>(timeKeys, &timeIn, &timeOut)
                   : MakeCurve<float, float>(timeKeys, nullptr, nullptr);

    const KeyData* in = valueBezier ? &valueIn : nullptr;
    const KeyData* out = valueBezier ? &valueOut : nullptr;
    std::shared_ptr<CurveBase> valueCurve;
    SceneObjectTrackType type = SceneObjectTrackType::kScalar;
    switch (valueKeys.arraySize) {
        case 0:
        case 1:
            valueCurve = MakeCurve<float, float>(valueKeys, in, out);
            type = SceneObjectTrackType::kScalar;
            break;
        case 3:
            valueCurve = MakeCurve<Vector3f, Vector3f>(valueKeys, in, out);
            type = SceneObjectTrackType::kVector3;
            break;
        case 4:
            valueCurve = MakeCurve<Quaternion<float>, float>(valueKeys, in, out);
            type = SceneObjectTrackType::kQuoternion;
            break;
        case 16:
            valueCurve = MakeCurve<Matrix4X4f, float>(valueKeys, in, out);
            type = SceneObjectTrackType::kMatrix;
            break;
        default:
            return false;
    }

    if (keyCount > 0 && timeKeys.values[keyCount - 1] > clip.GetEndTime()) {
        clip.SetEndTime(timeKeys.values[keyCount - 1]);
    }

    auto sceneTrack = std::make_shared<SceneObjectTrack>(
        transform, timeCurve, valueCurve, type);
    clip.AddTrack(sceneTrack);
    return true;
}
}  // namespace

std::unique_ptr<Scene> OgexParser::Parse(const std::string& buf) {
    m_bUpIsYAxis = false;
    m_bInvalid = false;

    if (!IsAcceptableText(buf)) return nullptr;

    OGEX::OpenGexDataDescription openGexDataDescription;

    const ODDL::DataResult result =
        openGexDataDescription.ProcessText(buf.c_str());
    if (result != ODDL::kDataOkay) {
        Report("Invalid OpenGEX data (error " + std::to_string(result) + ")");
        return nullptr;
    }

    auto pScene = std::make_unique<Scene>("OGEX Scene");
    const ODDL::Structure* structure =
        openGexDataDescription.GetRootStructure()->GetFirstSubnode();

    while (structure) {
        ConvertOddlStructureToSceneNode(*structure, pScene->SceneGraph,
                                        *pScene);
        if (m_bInvalid) return nullptr;

        structure = structure->Next();
    }

    return pScene;
}

void OgexParser::Fail(const std::string& message) {
    Report(message);
    m_bInvalid = true;
}

void OgexParser::ConvertOddlStructureToSceneNode(
    const ODDL::Structure& structure, std::shared_ptr<BaseSceneNode>& base_node,
    Scene& scene) {
    std::shared_ptr<BaseSceneNode> node;

    switch (structure.GetStructureType()) {
        case OGEX::kStructureMetric: {
            const auto* metric = Cast<OGEX::MetricStructure>(&structure);
            if (!metric) return Fail("invalid Metric structure");
            if (metric->GetMetricKey() == "up") {
                const auto* axis = Cast<ODDL::DataStructure<ODDL::StringDataType>>(
                    metric->GetFirstCoreSubnode());
                if (!axis || axis->GetDataElementCount() < 1) {
                    return Fail("Metric up requires a string value");
                }
                m_bUpIsYAxis = std::string_view(axis->GetDataElement(0)) == "y";
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
            scene.BoneNodes.emplace(structure.GetStructureName(), _node);
            node = _node;
        } break;
        case OGEX::kStructureGeometryNode: {
            const auto* _structure = Cast<OGEX::GeometryNodeStructure>(&structure);
            if (!_structure || !_structure->GetObjectStructure()) {
                return Fail("GeometryNode without a valid object reference");
            }
            const std::string key = structure.GetStructureName();
            auto _node = std::make_shared<SceneGeometryNode>(key);

            _node->SetVisibility(_structure->GetVisibleFlag());
            _node->SetIfCastShadow(_structure->GetShadowFlag());
            _node->SetIfMotionBlur(_structure->GetMotionBlurFlag());

            scene.GeometryNodes.emplace(key, _node);

            _node->AddSceneObjectRef(
                _structure->GetObjectStructure()->GetStructureName());

            const auto& materials = _structure->GetMaterialStructureArray();
            const auto materials_count = materials.GetElementCount();
            for (auto i = 0; i < materials_count; i++) {
                if (!materials[i]) {
                    return Fail("GeometryNode references a missing material");
                }
                _node->AddMaterialRef(materials[i]->GetStructureName());
            }

            scene.LUT_Name_GeometryNode.emplace(_structure->GetNodeName(),
                                                _node);

            node = _node;
        } break;
        case OGEX::kStructureLightNode: {
            const auto* _structure = Cast<OGEX::LightNodeStructure>(&structure);
            if (!_structure || !_structure->GetObjectStructure()) {
                return Fail("LightNode without a valid object reference");
            }
            auto _node =
                std::make_shared<SceneLightNode>(structure.GetStructureName());

            _node->SetIfCastShadow(_structure->GetShadowFlag());

            const std::string key =
                _structure->GetObjectStructure()->GetStructureName();
            _node->AddSceneObjectRef(key);

            scene.LightNodes.emplace(key, _node);

            node = _node;
        } break;
        case OGEX::kStructureCameraNode: {
            const auto* _structure = Cast<OGEX::CameraNodeStructure>(&structure);
            if (!_structure || !_structure->GetObjectStructure()) {
                return Fail("CameraNode without a valid object reference");
            }
            auto _node =
                std::make_shared<SceneCameraNode>(structure.GetStructureName());

            const std::string key =
                _structure->GetObjectStructure()->GetStructureName();
            _node->AddSceneObjectRef(key);

            scene.CameraNodes.emplace(key, _node);

            for (auto* extension = _structure->GetFirstExtensionSubnode();
                 extension; extension = extension->Next()) {
                const auto* cameraExtension =
                    Cast<OGEX::ExtensionStructure>(extension);
                if (cameraExtension &&
                    cameraExtension->GetApplicationString() ==
                        "MyGameEngine" &&
                    cameraExtension->GetTypeString() == "camera_target") {
                    const auto* data =
                        Cast<ODDL::DataStructure<ODDL::FloatDataType>>(
                            cameraExtension->GetFirstCoreSubnode());
                    if (!data || data != cameraExtension->GetLastCoreSubnode() ||
                        data->GetDataElementCount() != 3 ||
                        !AllFinite(&data->GetDataElement(0), 3)) {
                        return Fail(
                            "camera_target requires exactly three finite "
                            "world-space floats");
                    }
                    _node->SetTarget(Vector3f{data->GetDataElement(0),
                                              data->GetDataElement(1),
                                              data->GetDataElement(2)});
                    break;
                }
            }

            node = _node;
        } break;
        case OGEX::kStructureGeometryObject: {
            const auto* _structure =
                Cast<OGEX::GeometryObjectStructure>(&structure);
            if (!_structure) return Fail("invalid GeometryObject structure");
            const std::string key = structure.GetStructureName();
            auto _object = std::make_shared<SceneObjectGeometry>();

            _object->SetVisibility(_structure->GetVisibleFlag());
            _object->SetIfCastShadow(_structure->GetShadowFlag());
            _object->SetIfMotionBlur(_structure->GetMotionBlurFlag());

            SceneObjectPhysicsProperties physics;
            for (const ODDL::Structure* extension =
                     _structure->GetFirstExtensionSubnode();
                 extension; extension = extension->Next()) {
                const auto* _extension =
                    Cast<OGEX::ExtensionStructure>(extension);
                if (_extension &&
                    _extension->GetApplicationString() == "MyGameEngine" &&
                    !ParseGeometryExtension(*_extension, *_object, physics)) {
                    return Fail("invalid " +
                                ToStr(_extension->GetTypeString()) +
                                " extension on geometry " + key);
                }
            }
            _object->SetPhysicsProperties(physics);

            const ODDL::Map<OGEX::MeshStructure>* meshes =
                _structure->GetMeshMap();
            const int32_t count = meshes->GetElementCount();
            for (int32_t i = 0; i < count; i++) {
                std::shared_ptr<SceneObjectMesh> mesh;
                if (!ParseMesh(*(*meshes)[i], mesh)) {
                    return Fail("invalid mesh data in geometry " + key);
                }
                if (mesh) _object->AddMesh(std::move(mesh));
            }

            scene.Geometries[key] = _object;
        }
            return;
        case OGEX::kStructureTransform: {
            const auto* _structure = Cast<OGEX::TransformStructure>(&structure);
            if (!_structure || !base_node) {
                return Fail("invalid Transform structure");
            }
            const bool object_flag = _structure->GetObjectFlag();
            const std::string key = _structure->GetStructureName();
            const int32_t count = _structure->GetTransformCount();
            for (int32_t index = 0; index < count; index++) {
                const float* data = _structure->GetTransform(index);
                if (!data || !AllFinite(data, 16)) {
                    return Fail("Transform requires 16 finite floats");
                }
                Matrix4X4f matrix;
                matrix = data;
                base_node->AppendTransform(
                    key.c_str(), std::make_shared<SceneObjectTransform>(
                                     matrix, object_flag));
            }
        }
            return;
        case OGEX::kStructureTranslation: {
            const auto* _structure = Cast<OGEX::TranslationStructure>(&structure);
            if (!_structure || !base_node) {
                return Fail("invalid Translation structure");
            }
            const bool object_flag = _structure->GetObjectFlag();
            const std::string kind = ToStr(_structure->GetTranslationKind());
            const float* data = _structure->GetTranslation();
            std::shared_ptr<SceneObjectTranslation> translation;
            if (kind == "xyz" && data && AllFinite(data, 3)) {
                translation = std::make_shared<SceneObjectTranslation>(
                    data[0], data[1], data[2], object_flag);
            } else if ((kind == "x" || kind == "y" || kind == "z") && data &&
                       AllFinite(data, 1)) {
                translation = std::make_shared<SceneObjectTranslation>(
                    kind[0], data[0], object_flag);
            } else {
                return Fail("invalid Translation kind or value");
            }
            base_node->AppendTransform(_structure->GetStructureName(),
                                       std::move(translation));
        }
            return;
        case OGEX::kStructureRotation: {
            const auto* _structure = Cast<OGEX::RotationStructure>(&structure);
            if (!_structure || !base_node) {
                return Fail("invalid Rotation structure");
            }
            const bool object_flag = _structure->GetObjectFlag();
            const std::string kind = ToStr(_structure->GetRotationKind());
            const float* data = _structure->GetRotation();
            std::shared_ptr<SceneObjectRotation> rotation;
            if ((kind == "x" || kind == "y" || kind == "z") && data &&
                AllFinite(data, 1)) {
                rotation = std::make_shared<SceneObjectRotation>(
                    kind[0], data[0], object_flag);
            } else if (kind == "axis" && data && AllFinite(data, 4)) {
                rotation = std::make_shared<SceneObjectRotation>(
                    Vector3f({data[0], data[1], data[2]}), data[3],
                    object_flag);
            } else if (kind == "quaternion" && data && AllFinite(data, 4)) {
                rotation = std::make_shared<SceneObjectRotation>(
                    Quaternion<float>({data[0], data[1], data[2], data[3]}),
                    object_flag);
            } else {
                return Fail("invalid Rotation kind or value");
            }
            base_node->AppendTransform(_structure->GetStructureName(),
                                       std::move(rotation));
        }
            return;
        case OGEX::kStructureScale: {
            const auto* _structure = Cast<OGEX::ScaleStructure>(&structure);
            if (!_structure || !base_node) {
                return Fail("invalid Scale structure");
            }
            const bool object_flag = _structure->GetObjectFlag();
            const std::string kind = ToStr(_structure->GetScaleKind());
            const float* data = _structure->GetScale();
            std::shared_ptr<SceneObjectScale> scale;
            if ((kind == "x" || kind == "y" || kind == "z") && data &&
                AllFinite(data, 1)) {
                scale = std::make_shared<SceneObjectScale>(kind[0], data[0],
                                                           object_flag);
            } else if (kind == "xyz" && data && AllFinite(data, 3)) {
                scale = std::make_shared<SceneObjectScale>(
                    data[0], data[1], data[2], object_flag);
            } else {
                return Fail("invalid Scale kind or value");
            }
            base_node->AppendTransform(_structure->GetStructureName(),
                                       std::move(scale));
        }
            return;
        case OGEX::kStructureMaterial: {
            const auto* _structure = Cast<OGEX::MaterialStructure>(&structure);
            if (!_structure) return Fail("invalid Material structure");
            auto material = std::make_shared<SceneObjectMaterial>();
            material->SetName(ToStr(_structure->GetMaterialName()));

            if (!ParseAttributeChildren(_structure->GetFirstCoreSubnode(),
                                        *material)) {
                return Fail("invalid material attribute");
            }
            scene.Materials[_structure->GetStructureName()] = material;
        }
            return;
        case OGEX::kStructureLightObject: {
            const auto* _structure = Cast<OGEX::LightObjectStructure>(&structure);
            if (!_structure) return Fail("invalid LightObject structure");
            const std::string_view type(
                static_cast<const char*>(_structure->GetTypeString()));
            std::shared_ptr<SceneObjectLight> light;

            if (type == "infinite") {
                light = std::make_shared<SceneObjectInfiniteLight>();
            } else if (type == "point") {
                light = std::make_shared<SceneObjectOmniLight>();
            } else if (type == "spot") {
                light = std::make_shared<SceneObjectSpotLight>();
            } else if (type == "area") {
                light = std::make_shared<SceneObjectAreaLight>();
            } else {
                return Fail("unsupported light type " + std::string(type));
            }

            light->SetIfCastShadow(_structure->GetShadowFlag());

            if (!ParseAttributeChildren(_structure->GetFirstCoreSubnode(),
                                        *light)) {
                return Fail("invalid light attribute");
            }

            for (const ODDL::Structure* sub = _structure->GetFirstCoreSubnode();
                 sub; sub = sub->Next()) {
                if (sub->GetStructureType() != OGEX::kStructureAtten) continue;
                const auto* atten = Cast<OGEX::AttenStructure>(sub);
                if (!atten || !ParseAtten(*atten, *light)) {
                    return Fail("invalid light attenuation");
                }
            }

            for (const ODDL::Structure* extension =
                     _structure->GetFirstExtensionSubnode();
                 extension; extension = extension->Next()) {
                const auto* _extension =
                    Cast<OGEX::ExtensionStructure>(extension);
                if (_extension && !ParseLightExtension(*_extension, *light)) {
                    return Fail("invalid light extension");
                }
            }

            scene.Lights[_structure->GetStructureName()] = light;
        }
            return;
        case OGEX::kStructureCameraObject: {
            const auto* _structure = Cast<OGEX::CameraObjectStructure>(&structure);
            if (!_structure) return Fail("invalid CameraObject structure");
            auto camera = std::make_shared<SceneObjectPerspectiveCamera>();

            if (!ParseAttributeChildren(_structure->GetFirstCoreSubnode(),
                                        *camera)) {
                return Fail("invalid camera attribute");
            }
            scene.Cameras[_structure->GetStructureName()] = camera;
        }
            return;
        case OGEX::kStructureAnimation: {
            const auto* _structure = Cast<OGEX::AnimationStructure>(&structure);
            if (!_structure || !base_node) {
                return Fail("invalid Animation structure");
            }
            const auto clip_index = _structure->GetClipIndex();
            auto clip = std::make_shared<SceneObjectAnimationClip>(clip_index);

            for (const ODDL::Structure* sub = _structure->GetFirstCoreSubnode();
                 sub; sub = sub->Next()) {
                if (sub->GetStructureType() != OGEX::kStructureTrack) continue;
                const auto* track = Cast<OGEX::TrackStructure>(sub);
                if (!track || !ParseTrack(*track, *base_node, *clip)) {
                    return Fail("invalid animation track");
                }
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
        if (m_bInvalid) return;

        sub_structure = sub_structure->Next();
    }

    base_node->AppendChild(std::move(node));
}
