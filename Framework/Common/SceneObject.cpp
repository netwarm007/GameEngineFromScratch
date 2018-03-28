#include "SceneObject.hpp"
#include "quickhull.hpp"

using namespace std;

namespace My {
    ostream& operator<<(ostream& out, SceneObjectType type)
    {
        int32_t n = static_cast<int32_t>(type);
        n = endian_net_unsigned_int<int32_t>(n);
        char* c = reinterpret_cast<char*>(&n);
         
        for (size_t i = 0; i < sizeof(int32_t); i++) {
            out << *c++;
        }

        return out;
    }

    ostream& operator<<(ostream& out, IndexDataType type)
    {
        int32_t n = static_cast<int32_t>(type);
        n = endian_net_unsigned_int<int32_t>(n);
        char* c = reinterpret_cast<char*>(&n);
         
        for (size_t i = 0; i < sizeof(int32_t); i++) {
            out << *c++;
        }

        return out;
    }

    ostream& operator<<(ostream& out, VertexDataType type)
    {
        int32_t n = static_cast<int32_t>(type);
        n = endian_net_unsigned_int<int32_t>(n);
        char* c = reinterpret_cast<char*>(&n);
         
        for (size_t i = 0; i < sizeof(int32_t); i++) {
            out << *c++;
        }

        return out;
    }

    ostream& operator<<(ostream& out, PrimitiveType type)
    {
        int32_t n = static_cast<int32_t>(type);
        n = endian_net_unsigned_int<int32_t>(n);
        char* c = reinterpret_cast<char*>(&n);
         
        for (size_t i = 0; i < sizeof(int32_t); i++) {
            out << *c++;
        }

        return out;
    }
  
    ostream& operator<<(ostream& out, CurveType type)
    {
        int32_t n = static_cast<int32_t>(type);
        n = endian_net_unsigned_int<int32_t>(n);
        char* c = reinterpret_cast<char*>(&n);
         
        for (size_t i = 0; i < sizeof(int32_t); i++) {
            out << *c++;
        }

        return out;
    }

	ostream& operator<<(ostream& out, const BaseSceneObject& obj)
	{
		out << "SceneObject" << endl;
		out << "-----------" << endl;
		out << "GUID: " << obj.m_Guid << endl;
		out << "Type: " << obj.m_Type << endl;

		return out;
	}

	ostream& operator<<(ostream& out, const SceneObjectVertexArray& obj)
	{
		out << "Attribute: " << obj.m_strAttribute << endl;
		out << "Morph Target Index: 0x" << obj.m_nMorphTargetIndex << endl;
		out << "Data Type: " << obj.m_DataType << endl;
		out << "Data Size: 0x" << obj.m_szData << endl;
		out << "Data: ";
		for(size_t i = 0; i < obj.m_szData; i++)
		{
			out << *(reinterpret_cast<const float*>(obj.m_pData) + i) << ' ';;
		}

		return out;
	}

	ostream& operator<<(ostream& out, const SceneObjectIndexArray& obj)
	{
		out << "Material Index: 0x" << obj.m_nMaterialIndex << endl;
		out << "Restart Index: 0x" << obj.m_szRestartIndex << endl;
		out << "Data Type: " << obj.m_DataType << endl;
		out << "Data Size: 0x" << obj.m_szData << endl;
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

	ostream& operator<<(ostream& out, const SceneObjectMesh& obj)
	{
		out << static_cast<const BaseSceneObject&>(obj) << endl;
		out << "Primitive Type: " << obj.m_PrimitiveType << endl;
		out << "This mesh contains 0x" << obj.m_VertexArray.size() << " vertex properties." << endl;
		for (size_t i = 0; i < obj.m_VertexArray.size(); i++) {
			out << obj.m_VertexArray[i] << endl;
		}
		out << "This mesh contains 0x" << obj.m_IndexArray.size() << " index arrays." << endl;
		for (size_t i = 0; i < obj.m_IndexArray.size(); i++) {
			out << obj.m_IndexArray[i] << endl;
		}

		return out;
	}

	ostream& operator<<(ostream& out, const SceneObjectTexture& obj)
	{
		out << static_cast<const BaseSceneObject&>(obj) << endl;
		out << "Coord Index: " << obj.m_nTexCoordIndex << endl;
		out << "Name: " << obj.m_Name << endl;
		if (obj.m_pImage)
			out << "Image: " << *obj.m_pImage << endl;

		return out;
	}

	ostream& operator<<(ostream& out, const SceneObjectMaterial& obj)
	{
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

	ostream& operator<<(ostream& out, const SceneObjectGeometry& obj)
	{
		auto count = obj.m_Mesh.size();
		for(decltype(count) i = 0; i < count; i++) {
			out << "Mesh: " << *obj.m_Mesh[i] << endl;
		}

		return out;
	}

	ostream& operator<<(ostream& out, const SceneObjectLight& obj)
	{
		out << static_cast<const BaseSceneObject&>(obj) << endl;
		out << "Color: " << obj.m_LightColor << endl;
		out << "Intensity: " << obj.m_fIntensity << endl;
		out << "Cast Shadows: " << obj.m_bCastShadows << endl;

		return out;
	}

	ostream& operator<<(ostream& out, const SceneObjectOmniLight& obj)
	{
		out << static_cast<const SceneObjectLight&>(obj) << endl;
		out << "Light Type: Omni" << endl;

		return out;
	}

	ostream& operator<<(ostream& out, const SceneObjectSpotLight& obj)
	{
		out << static_cast<const SceneObjectLight&>(obj) << endl;
		out << "Light Type: Spot" << endl;
		out << "Cone Angle: " << obj.m_fConeAngle << endl;
		out << "Penumbra Angle: " << obj.m_fPenumbraAngle << endl;

		return out;
	}

	ostream& operator<<(ostream& out, const SceneObjectInfiniteLight& obj)
	{
		out << static_cast<const SceneObjectLight&>(obj) << endl;
		out << "Light Type: Infinite" << endl;

		return out;
	}

	ostream& operator<<(ostream& out, const SceneObjectCamera& obj)
	{
		out << static_cast<const BaseSceneObject&>(obj) << endl;
		out << "Aspect: " << obj.m_fAspect << endl;
		out << "Near Clip Distance: " << obj.m_fNearClipDistance << endl;
		out << "Far Clip Distance: " << obj.m_fFarClipDistance << endl;

		return out;
	}

	ostream& operator<<(ostream& out, const SceneObjectOrthogonalCamera& obj)
	{
		out << static_cast<const SceneObjectCamera&>(obj) << endl;
		out << "Camera Type: Orthogonal" << endl;

		return out;
	}


	ostream& operator<<(ostream& out, const SceneObjectPerspectiveCamera& obj)
	{
		out << static_cast<const SceneObjectCamera&>(obj) << endl;
		out << "Camera Type: Perspective" << endl;
		out << "FOV: " << obj.m_fFov<< endl;

		return out;
	}

	ostream& operator<<(ostream& out, const SceneObjectTransform& obj)
	{
		out << "Transform Matrix: " << obj.m_matrix << endl;
		out << "Is Object Local: " << obj.m_bSceneObjectOnly << endl;

		return out;
	}

    ostream& operator<<(ostream& out, const SceneObjectTrack& obj)
	{
		out << "Animation Track: " << endl;
		out << "Time: " << obj.m_Time->GetCurveType() << endl;
		out << "Value: " << obj.m_Value->GetCurveType() << endl;
		out << "Transform: " << *obj.m_pTransform << endl;

		return out;
	}

    ostream& operator<<(ostream& out, const SceneObjectAnimationClip& obj)
	{
		out << "Animation Clip: " << obj.m_nIndex << endl;
		out << "Num of Track(s): " << obj.m_Tracks.size() << endl;
		for (auto track : obj.m_Tracks)
		{
			out << *track;
		}

		return out;
	}

    float DefaultAttenFunc(float intensity, float distance)
    {
        return intensity / pow(1 + distance, 2.0f);
    }

	BoundingBox SceneObjectMesh::GetBoundingBox() const
	{
		Vector3f bbmin (numeric_limits<float>::max());
		Vector3f bbmax (numeric_limits<float>::lowest());
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
							bbmin[0] = (bbmin[0] < vertex->data[0])? bbmin[0] : vertex->data[0];
							bbmin[1] = (bbmin[1] < vertex->data[1])? bbmin[1] : vertex->data[1];
							bbmin[2] = (bbmin[2] < vertex->data[2])? bbmin[2] : vertex->data[2];
							bbmax[0] = (bbmax[0] > vertex->data[0])? bbmax[0] : vertex->data[0];
							bbmax[1] = (bbmax[1] > vertex->data[1])? bbmax[1] : vertex->data[1];
							bbmax[2] = (bbmax[2] > vertex->data[2])? bbmax[2] : vertex->data[2];
							break;
						}
						case VertexDataType::kVertexDataTypeDouble3:
						{
							const Vector3* vertex = reinterpret_cast<const Vector3*>(data) + i;
							bbmin[0] = static_cast<float>((bbmin[0] < vertex->data[0])? bbmin[0] : vertex->data[0]);
							bbmin[1] = static_cast<float>((bbmin[1] < vertex->data[1])? bbmin[1] : vertex->data[1]);
							bbmin[2] = static_cast<float>((bbmin[2] < vertex->data[2])? bbmin[2] : vertex->data[2]);
							bbmax[0] = static_cast<float>((bbmax[0] > vertex->data[0])? bbmax[0] : vertex->data[0]);
							bbmax[1] = static_cast<float>((bbmax[1] > vertex->data[1])? bbmax[1] : vertex->data[1]);
							bbmax[2] = static_cast<float>((bbmax[2] > vertex->data[2])? bbmax[2] : vertex->data[2]);
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

	ConvexHull SceneObjectMesh::GetConvexHull() const
	{
		ConvexHull hull;

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
							hull.AddPoint(*vertex);
							break;
						}
						case VertexDataType::kVertexDataTypeDouble3:
						{
							const Vector3* vertex = reinterpret_cast<const Vector3*>(data) + i;
							hull.AddPoint(*vertex);
							break;
						}
						default:
							assert(0);
					}
				}
			}
		}

		// calculate the convex hull
		hull.Iterate();

		return hull;
	}

	void SceneObjectTrack::Update(const float time_point)
	{
		if (m_pTransform)
		{
			float new_val = m_Value->Interpolate(m_Time->Reverse(time_point));
			switch (m_pTransform->GetType())
			{
				case SceneObjectType::kSceneObjectTypeTranslate:
					{
						auto pObj = dynamic_pointer_cast<SceneObjectTranslation>(m_pTransform);
						pObj->Update(new_val);
					}
					break;
				case SceneObjectType::kSceneObjectTypeRotate:
					{
						auto pObj = dynamic_pointer_cast<SceneObjectRotation>(m_pTransform);
						pObj->Update(new_val);
					}
					break;
				case SceneObjectType::kSceneObjectTypeScale:
					{
						auto pObj = dynamic_pointer_cast<SceneObjectScale>(m_pTransform);
						pObj->Update(new_val);
					}
					break;
				default:
					assert(0);
			}
		}
	}

    void SceneObjectAnimationClip::AddTrack(shared_ptr<SceneObjectTrack>& track)
	{
		m_Tracks.push_back(track);
	}

	void SceneObjectAnimationClip::Update(const float time_point)
	{
		for (auto track : m_Tracks)
		{
			track->Update(time_point);
		}
	}
}
