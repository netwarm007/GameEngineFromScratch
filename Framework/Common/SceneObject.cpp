#include "SceneObject.hpp"

#include "quickhull.hpp"

using namespace std;

namespace My {
ostream& operator<<(ostream& out, SceneObjectType type) {
    auto n = static_cast<int32_t>(type);
    n = endian_net_unsigned_int<int32_t>(n);
    char* c = reinterpret_cast<char*>(&n);

    for (size_t i = 0; i < sizeof(int32_t); i++) {
        out << *c++;
    }

    return out;
}

ostream& operator<<(ostream& out, IndexDataType type) {
    auto n = static_cast<int32_t>(type);
    n = endian_net_unsigned_int<int32_t>(n);
    char* c = reinterpret_cast<char*>(&n);

    for (size_t i = 0; i < sizeof(int32_t); i++) {
        out << *c++;
    }

    return out;
}

ostream& operator<<(ostream& out, VertexDataType type) {
    auto n = static_cast<int32_t>(type);
    n = endian_net_unsigned_int<int32_t>(n);
    char* c = reinterpret_cast<char*>(&n);

    for (size_t i = 0; i < sizeof(int32_t); i++) {
        out << *c++;
    }

    return out;
}

ostream& operator<<(ostream& out, PrimitiveType type) {
    auto n = static_cast<int32_t>(type);
    n = endian_net_unsigned_int<int32_t>(n);
    char* c = reinterpret_cast<char*>(&n);

    for (size_t i = 0; i < sizeof(int32_t); i++) {
        out << *c++;
    }

    return out;
}

ostream& operator<<(ostream& out, CurveType type) {
    auto n = static_cast<int32_t>(type);
    n = endian_net_unsigned_int<int32_t>(n);
    char* c = reinterpret_cast<char*>(&n);

    for (size_t i = 0; i < sizeof(int32_t); i++) {
        out << *c++;
    }

    return out;
}

ostream& operator<<(ostream& out, const BaseSceneObject& obj) {
    out << "SceneObject" << endl;
    out << "-----------" << endl;
    out << "GUID: " << obj.m_Guid << endl;
    out << "Type: " << obj.m_Type << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectVertexArray& obj) {
    out << "Attribute: " << obj.m_strAttribute << endl;
    out << "Morph Target Index: 0x" << obj.m_nMorphTargetIndex << endl;
    out << "Data Type: " << obj.m_DataType << endl;
    out << "Data Size: 0x" << obj.m_szData << endl;
    out << "Data: ";
    for (size_t i = 0; i < obj.m_szData; i++) {
        out << *(reinterpret_cast<const float*>(obj.m_pData) + i) << ' ';
        ;
    }

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectIndexArray& obj) {
    out << "Material Index: 0x" << obj.m_nMaterialIndex << endl;
    out << "Restart Index: 0x" << obj.m_szRestartIndex << endl;
    out << "Data Type: " << obj.m_DataType << endl;
    out << "Data Size: 0x" << obj.m_szData << endl;
    out << "Data: ";
    for (size_t i = 0; i < obj.m_szData; i++) {
        switch (obj.m_DataType) {
            case IndexDataType::kIndexDataTypeInt8:
                out << "0x"
                    << *(reinterpret_cast<const uint8_t*>(obj.m_pData) + i)
                    << ' ';
                ;
                break;
            case IndexDataType::kIndexDataTypeInt16:
                out << "0x"
                    << *(reinterpret_cast<const uint16_t*>(obj.m_pData) + i)
                    << ' ';
                ;
                break;
            case IndexDataType::kIndexDataTypeInt32:
                out << "0x"
                    << *(reinterpret_cast<const uint32_t*>(obj.m_pData) + i)
                    << ' ';
                ;
                break;
            case IndexDataType::kIndexDataTypeInt64:
                out << "0x"
                    << *(reinterpret_cast<const uint64_t*>(obj.m_pData) + i)
                    << ' ';
                ;
                break;
            default:;
        }
    }

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectMesh& obj) {
    out << static_cast<const BaseSceneObject&>(obj) << endl;
    out << "Primitive Type: " << obj.m_PrimitiveType << endl;
    out << "This mesh contains 0x" << obj.m_VertexArray.size()
        << " vertex properties." << endl;
    for (const auto& vertex : obj.m_VertexArray) {
        out << vertex << endl;
    }
    out << "This mesh contains 0x" << obj.m_IndexArray.size()
        << " index arrays." << endl;
    for (const auto& index : obj.m_IndexArray) {
        out << index << endl;
    }

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectTexture& obj) {
    out << static_cast<const BaseSceneObject&>(obj) << endl;
    out << "Coord Index: " << obj.m_nTexCoordIndex << endl;
    out << "Name: " << obj.m_Name << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectMaterial& obj) {
    out << static_cast<const BaseSceneObject&>(obj) << endl;
    out << "Name: " << obj.m_Name << endl;
    out << "Albedo: " << obj.m_BaseColor << endl;
    out << "Metallic: " << obj.m_Metallic << endl;
    out << "Roughness: " << obj.m_Roughness << endl;
    out << "Normal: " << obj.m_Normal << endl;
    out << "Specular: " << obj.m_Specular << endl;
    out << "Ambient Occlusion:: " << obj.m_AmbientOcclusion << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectGeometry& obj) {
    auto count = obj.m_Mesh.size();
    for (decltype(count) i = 0; i < count; i++) {
        out << "Mesh: " << *obj.m_Mesh[i] << endl;
    }

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectLight& obj) {
    out << static_cast<const BaseSceneObject&>(obj) << endl;
    out << "Color: " << obj.m_LightColor << endl;
    out << "Intensity: " << obj.m_fIntensity << endl;
    out << "Cast Shadows: " << obj.m_bCastShadows << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectOmniLight& obj) {
    out << static_cast<const SceneObjectLight&>(obj) << endl;
    out << "Light Type: Omni" << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectSpotLight& obj) {
    out << static_cast<const SceneObjectLight&>(obj) << endl;
    out << "Light Type: Spot" << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectInfiniteLight& obj) {
    out << static_cast<const SceneObjectLight&>(obj) << endl;
    out << "Light Type: Infinite" << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectAreaLight& obj) {
    out << static_cast<const SceneObjectLight&>(obj) << endl;
    out << "Light Type: Area" << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectCamera& obj) {
    out << static_cast<const BaseSceneObject&>(obj) << endl;
    out << "Aspect: " << obj.m_fAspect << endl;
    out << "Near Clip Distance: " << obj.m_fNearClipDistance << endl;
    out << "Far Clip Distance: " << obj.m_fFarClipDistance << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectOrthogonalCamera& obj) {
    out << static_cast<const SceneObjectCamera&>(obj) << endl;
    out << "Camera Type: Orthogonal" << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectPerspectiveCamera& obj) {
    out << static_cast<const SceneObjectCamera&>(obj) << endl;
    out << "Camera Type: Perspective" << endl;
    out << "FOV: " << obj.m_fFov << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectTransform& obj) {
    out << "Transform Matrix: " << obj.m_matrix << endl;
    out << "Is Object Local: " << obj.m_bSceneObjectOnly << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectTrack& obj) {
    out << "Animation Track: " << endl;
    out << "Time: " << obj.m_Time->GetCurveType() << endl;
    out << "Value: " << obj.m_Value->GetCurveType() << endl;
    out << "Transform: " << *obj.m_pTransform << endl;

    return out;
}

ostream& operator<<(ostream& out, const SceneObjectAnimationClip& obj) {
    out << "Animation Clip: " << obj.m_nIndex << endl;
    out << "Num of Track(s): " << obj.m_Tracks.size() << endl;
    for (const auto& track : obj.m_Tracks) {
        out << *track;
    }

    return out;
}

float DefaultAttenFunc(float intensity, float distance) {
    return intensity / pow(1 + distance, 2.0f);
}
}  // namespace My
