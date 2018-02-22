#include "SceneObject.hpp"

using namespace std;

namespace My {
    std::ostream& operator<<(std::ostream& out, SceneObjectType type)
    {
        int32_t n = static_cast<int32_t>(type);
        n = endian_net_unsigned_int<int32_t>(n);
        char* c = reinterpret_cast<char*>(&n);
         
        for (size_t i = 0; i < sizeof(int32_t); i++) {
            out << *c++;
        }

        return out;
    }

    std::ostream& operator<<(std::ostream& out, IndexDataType type)
    {
        int32_t n = static_cast<int32_t>(type);
        n = endian_net_unsigned_int<int32_t>(n);
        char* c = reinterpret_cast<char*>(&n);
         
        for (size_t i = 0; i < sizeof(int32_t); i++) {
            out << *c++;
        }

        return out;
    }

    std::ostream& operator<<(std::ostream& out, VertexDataType type)
    {
        int32_t n = static_cast<int32_t>(type);
        n = endian_net_unsigned_int<int32_t>(n);
        char* c = reinterpret_cast<char*>(&n);
         
        for (size_t i = 0; i < sizeof(int32_t); i++) {
            out << *c++;
        }

        return out;
    }

    std::ostream& operator<<(std::ostream& out, PrimitiveType type)
    {
        int32_t n = static_cast<int32_t>(type);
        n = endian_net_unsigned_int<int32_t>(n);
        char* c = reinterpret_cast<char*>(&n);
         
        for (size_t i = 0; i < sizeof(int32_t); i++) {
            out << *c++;
        }

        return out;
    }
  
	std::ostream& operator<<(std::ostream& out, const BaseSceneObject& obj)
	{
		out << "SceneObject" << std::endl;
		out << "-----------" << std::endl;
		out << "GUID: " << obj.m_Guid << std::endl;
		out << "Type: " << obj.m_Type << std::endl;

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectVertexArray& obj)
	{
		out << "Attribute: " << obj.m_strAttribute << std::endl;
		out << "Morph Target Index: 0x" << obj.m_nMorphTargetIndex << std::endl;
		out << "Data Type: " << obj.m_DataType << std::endl;
		out << "Data Size: 0x" << obj.m_szData << std::endl;
		out << "Data: ";
		for(size_t i = 0; i < obj.m_szData; i++)
		{
			out << *(reinterpret_cast<const float*>(obj.m_pData) + i) << ' ';;
		}

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectIndexArray& obj)
	{
		out << "Material Index: 0x" << obj.m_nMaterialIndex << std::endl;
		out << "Restart Index: 0x" << obj.m_szRestartIndex << std::endl;
		out << "Data Type: " << obj.m_DataType << std::endl;
		out << "Data Size: 0x" << obj.m_szData << std::endl;
		out << "Data: ";
		for(size_t i = 0; i < obj.m_szData; i++)
		{
			switch(obj.m_DataType)
			{
				case IndexDataType::kIndexDataTypeInt8:
					out << "0x" << *(reinterpret_cast<const uint8_t*>(obj.m_pData) + i) << ' ';;
					break;
				case IndexDataType::kIndexDataTypeInt16:
					out << "0x" << *(reinterpret_cast<const uint16_t*>(obj.m_pData) + i) << ' ';;
					break;
				case IndexDataType::kIndexDataTypeInt32:
					out << "0x" << *(reinterpret_cast<const uint32_t*>(obj.m_pData) + i) << ' ';;
					break;
				case IndexDataType::kIndexDataTypeInt64:
					out << "0x" << *(reinterpret_cast<const uint64_t*>(obj.m_pData) + i) << ' ';;
					break;
				default:
					;
			}
		}


		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectMesh& obj)
	{
		out << static_cast<const BaseSceneObject&>(obj) << std::endl;
		out << "Primitive Type: " << obj.m_PrimitiveType << std::endl;
		out << "This mesh contains 0x" << obj.m_VertexArray.size() << " vertex properties." << std::endl;
		for (size_t i = 0; i < obj.m_VertexArray.size(); i++) {
			out << obj.m_VertexArray[i] << std::endl;
		}
		out << "This mesh contains 0x" << obj.m_IndexArray.size() << " index arrays." << std::endl;
		for (size_t i = 0; i < obj.m_IndexArray.size(); i++) {
			out << obj.m_IndexArray[i] << std::endl;
		}

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectTexture& obj)
	{
		out << static_cast<const BaseSceneObject&>(obj) << std::endl;
		out << "Coord Index: " << obj.m_nTexCoordIndex << std::endl;
		out << "Name: " << obj.m_Name << std::endl;
		if (obj.m_pImage)
			out << "Image: " << *obj.m_pImage << std::endl;

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectMaterial& obj)
	{
		out << static_cast<const BaseSceneObject&>(obj) << std::endl;
		out << "Name: " << obj.m_Name << std::endl;
		out << "Albedo: " << obj.m_BaseColor << std::endl;
		out << "Metallic: " << obj.m_Metallic << std::endl;
		out << "Roughness: " << obj.m_Roughness << std::endl;
		out << "Normal: " << obj.m_Normal << std::endl;
		out << "Specular: " << obj.m_Specular << std::endl;
		out << "Ambient Occlusion:: " << obj.m_AmbientOcclusion << std::endl;

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectGeometry& obj)
	{
		auto count = obj.m_Mesh.size();
		for(decltype(count) i = 0; i < count; i++) {
			out << "Mesh: " << *obj.m_Mesh[i] << std::endl;
		}

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectLight& obj)
	{
		out << static_cast<const BaseSceneObject&>(obj) << std::endl;
		out << "Color: " << obj.m_LightColor << std::endl;
		out << "Intensity: " << obj.m_fIntensity << std::endl;
		out << "Cast Shadows: " << obj.m_bCastShadows << std::endl;

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectOmniLight& obj)
	{
		out << static_cast<const SceneObjectLight&>(obj) << std::endl;
		out << "Light Type: Omni" << std::endl;

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectSpotLight& obj)
	{
		out << static_cast<const SceneObjectLight&>(obj) << std::endl;
		out << "Light Type: Spot" << std::endl;
		out << "Cone Angle: " << obj.m_fConeAngle << std::endl;
		out << "Penumbra Angle: " << obj.m_fPenumbraAngle << std::endl;

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectInfiniteLight& obj)
	{
		out << static_cast<const SceneObjectLight&>(obj) << std::endl;
		out << "Light Type: Infinite" << std::endl;

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectCamera& obj)
	{
		out << static_cast<const BaseSceneObject&>(obj) << std::endl;
		out << "Aspect: " << obj.m_fAspect << std::endl;
		out << "Near Clip Distance: " << obj.m_fNearClipDistance << std::endl;
		out << "Far Clip Distance: " << obj.m_fFarClipDistance << std::endl;

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectOrthogonalCamera& obj)
	{
		out << static_cast<const SceneObjectCamera&>(obj) << std::endl;
		out << "Camera Type: Orthogonal" << std::endl;

		return out;
	}


	std::ostream& operator<<(std::ostream& out, const SceneObjectPerspectiveCamera& obj)
	{
		out << static_cast<const SceneObjectCamera&>(obj) << std::endl;
		out << "Camera Type: Perspective" << std::endl;
		out << "FOV: " << obj.m_fFov<< std::endl;

		return out;
	}

	std::ostream& operator<<(std::ostream& out, const SceneObjectTransform& obj)
	{
		out << "Transform Matrix: " << obj.m_matrix << std::endl;
		out << "Is Object Local: " << obj.m_bSceneObjectOnly << std::endl;

		return out;
	}


    float DefaultAttenFunc(float intensity, float distance)
    {
        return intensity / pow(1 + distance, 2.0f);
    }

	BoundingBox SceneObjectMesh::GetBoundingBox() const
	{
		Vector3f bbmin (numeric_limits<float>::max());
		Vector3f bbmax (numeric_limits<float>::min());
		auto count = m_VertexArray.size();
		for (auto n = 0; n < count; n++)
		{
			if (m_VertexArray[n].GetAttributeName() == "position")
			{
				auto data_type = m_VertexArray[n].GetDataType();
				auto vertices_count = m_VertexArray[n].GetVertexCount();	
				auto data = m_VertexArray[n].GetData();
				for (auto i = 0; i < vertices_count; i++)
				{
					switch(data_type) {
					case VertexDataType::kVertexDataTypeFloat3:
					{
						const Vector3f* vertex = reinterpret_cast<const Vector3f*>(data) + i;
						bbmin.x = (bbmin.x < vertex->x)? bbmin.x : vertex->x;
						bbmin.y = (bbmin.y < vertex->y)? bbmin.y : vertex->y;
						bbmin.z = (bbmin.z < vertex->z)? bbmin.z : vertex->z;
						bbmax.x = (bbmax.x > vertex->x)? bbmax.x : vertex->x;
						bbmax.y = (bbmax.y > vertex->y)? bbmax.y : vertex->y;
						bbmax.z = (bbmax.z > vertex->z)? bbmax.z : vertex->z;
						break;
					}
					case VertexDataType::kVertexDataTypeDouble3:
					{
						const Vector3* vertex = reinterpret_cast<const Vector3*>(data) + i;
						bbmin.x = (bbmin.x < vertex->x)? bbmin.x : vertex->x;
						bbmin.y = (bbmin.y < vertex->y)? bbmin.y : vertex->y;
						bbmin.z = (bbmin.z < vertex->z)? bbmin.z : vertex->z;
						bbmax.x = (bbmax.x > vertex->x)? bbmax.x : vertex->x;
						bbmax.y = (bbmax.y > vertex->y)? bbmax.y : vertex->y;
						bbmax.z = (bbmax.z > vertex->z)? bbmax.z : vertex->z;
						break;
					}
					default:
						assert(0);
					}
				}
			}
		}

		BoundingBox result;
		result.extent = (bbmax - bbmin) * 0.5f;
		result.centroid = (bbmax + bbmin) * 0.5f;

		return result;
	}
}
