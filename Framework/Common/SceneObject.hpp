#pragma once
#include <assert.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "Guid.hpp"
#include "Image.hpp"
#include "portable.hpp"
#include "geommath.hpp"

namespace My {
    namespace details {
        constexpr int32_t i32(const char* s, int32_t v) {
            return *s ? i32(s+1, v * 256 + *s) : v;
        }
    }

    constexpr int32_t operator "" _i32(const char* s, size_t) {
        return details::i32(s, 0);
    }

    ENUM(SceneObjectType) {
        kSceneObjectTypeMesh    =   "MESH"_i32,
        kSceneObjectTypeMaterial=   "MATL"_i32,
        kSceneObjectTypeTexture =   "TXTU"_i32,
        kSceneObjectTypeLight   =   "LGHT"_i32,
        kSceneObjectTypeCamera  =   "CAMR"_i32,
        kSceneObjectTypeAnimator=   "ANIM"_i32,
        kSceneObjectTypeClip    =   "CLIP"_i32,
        kSceneObjectTypeVertexArray   =   "VARR"_i32,
        kSceneObjectTypeIndexArray    =   "VARR"_i32,
        kSceneObjectTypeGeometry =  "GEOM"_i32,
    };

    std::ostream& operator<<(std::ostream& out, SceneObjectType type);

    using namespace xg;
    class BaseSceneObject
    {
        protected:
            Guid m_Guid;
            SceneObjectType m_Type;
        protected:
            // can only be used as base class
            BaseSceneObject(SceneObjectType type) : m_Type(type) { m_Guid = newGuid(); };
            BaseSceneObject(Guid& guid, SceneObjectType type) : m_Guid(guid), m_Type(type) {};
            BaseSceneObject(Guid&& guid, SceneObjectType type) : m_Guid(std::move(guid)), m_Type(type) {};
            BaseSceneObject(BaseSceneObject&& obj) : m_Guid(std::move(obj.m_Guid)), m_Type(obj.m_Type) {};
            BaseSceneObject& operator=(BaseSceneObject&& obj) { this->m_Guid = std::move(obj.m_Guid); this->m_Type = obj.m_Type; return *this; };
            virtual ~BaseSceneObject() {};
            
        private:
            // a type must be specified
            BaseSceneObject() = delete; 
            // can not be copied
            BaseSceneObject(BaseSceneObject& obj) = delete;
            BaseSceneObject& operator=(BaseSceneObject& obj) = delete;

        public:
            const Guid& GetGuid() const { return m_Guid; };
            const SceneObjectType GetType() const { return m_Type; };

        friend std::ostream& operator<<(std::ostream& out, const BaseSceneObject& obj);
    };

    ENUM(VertexDataType) {
        kVertexDataTypeFloat1    = "FLT1"_i32,
        kVertexDataTypeFloat2    = "FLT2"_i32,
        kVertexDataTypeFloat3    = "FLT3"_i32,
        kVertexDataTypeFloat4    = "FLT4"_i32,
        kVertexDataTypeDouble1   = "DUB1"_i32,
        kVertexDataTypeDouble2   = "DUB2"_i32,
        kVertexDataTypeDouble3   = "DUB3"_i32,
        kVertexDataTypeDouble4   = "DUB3"_i32
    };

    std::ostream& operator<<(std::ostream& out, VertexDataType type);

    class SceneObjectVertexArray 
    {
        protected:
            const std::string m_strAttribute;
            const uint32_t    m_nMorphTargetIndex;
            const VertexDataType m_DataType;

            const void*      m_pData;

            const size_t     m_szData;

        public:
            SceneObjectVertexArray(const char* attr = "", const uint32_t morph_index = 0, const VertexDataType data_type = VertexDataType::kVertexDataTypeFloat3, const void* data = nullptr, const size_t data_size = 0) : m_strAttribute(attr), m_nMorphTargetIndex(morph_index), m_DataType(data_type), m_pData(data), m_szData(data_size) {};
            SceneObjectVertexArray(SceneObjectVertexArray& arr) = default; 
            SceneObjectVertexArray(SceneObjectVertexArray&& arr) = default; 

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectVertexArray& obj);
    };

    ENUM(IndexDataType) {
        kIndexDataTypeInt8  = "I8  "_i32,
        kIndexDataTypeInt16 = "I16 "_i32,
        kIndexDataTypeInt32 = "I32 "_i32,
        kIndexDataTypeInt64 = "I64 "_i32,
    };

    std::ostream& operator<<(std::ostream& out, IndexDataType type);

    class SceneObjectIndexArray
    {
        protected:
            const uint32_t    m_nMaterialIndex;
            const size_t      m_szRestartIndex;
            const IndexDataType m_DataType;

            const void*       m_pData;

            const size_t      m_szData;

        public:
            SceneObjectIndexArray(const uint32_t material_index = 0, const size_t restart_index = 0, const IndexDataType data_type = IndexDataType::kIndexDataTypeInt16, const void* data = nullptr, const size_t data_size = 0) 
                : m_nMaterialIndex(material_index), m_szRestartIndex(restart_index), m_DataType(data_type), m_pData(data), m_szData(data_size) {};
            SceneObjectIndexArray(SceneObjectIndexArray& arr) = default;
            SceneObjectIndexArray(SceneObjectIndexArray&& arr) = default;

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectIndexArray& obj);
    };

	ENUM(PrimitiveType) {
		kPrimitiveTypeNone = "NONE"_i32,        ///< No particular primitive type.
		kPrimitiveTypePointList = "PLST"_i32,   ///< For N>=0, vertex N renders a point.
		kPrimitiveTypeLineList = "LLST"_i32,    ///< For N>=0, vertices [N*2+0, N*2+1] render a line.
		kPrimitiveTypeLineStrip = "LSTR"_i32,   ///< For N>=0, vertices [N, N+1] render a line.
		kPrimitiveTypeTriList = "TLST"_i32,     ///< For N>=0, vertices [N*3+0, N*3+1, N*3+2] render a triangle.
		kPrimitiveTypeTriFan = "TFAN"_i32,      ///< For N>=0, vertices [0, (N+1)%M, (N+2)%M] render a triangle, where M is the vertex count.
		kPrimitiveTypeTriStrip = "TSTR"_i32,    ///< For N>=0, vertices [N*2+0, N*2+1, N*2+2] and [N*2+2, N*2+1, N*2+3] render triangles.
		kPrimitiveTypePatch = "PACH"_i32,       ///< Used for tessellation.
		kPrimitiveTypeLineListAdjacency = "LLSA"_i32,       ///< For N>=0, vertices [N*4..N*4+3] render a line from [1, 2]. Lines [0, 1] and [2, 3] are adjacent to the rendered line.
		kPrimitiveTypeLineStripAdjacency = "LSTA"_i32,      ///< For N>=0, vertices [N+1, N+2] render a line. Lines [N, N+1] and [N+2, N+3] are adjacent to the rendered line.
		kPrimitiveTypeTriListAdjacency = "TLSA"_i32,        ///< For N>=0, vertices [N*6..N*6+5] render a triangle from [0, 2, 4]. Triangles [0, 1, 2] [4, 2, 3] and [5, 0, 4] are adjacent to the rendered triangle.
		kPrimitiveTypeTriStripAdjacency = "TSTA"_i32,       ///< For N>=0, vertices [N*4..N*4+6] render a triangle from [0, 2, 4] and [4, 2, 6]. Odd vertices Nodd form adjacent triangles with indices min(Nodd+1,Nlast) and max(Nodd-3,Nfirst).
		kPrimitiveTypeRectList = "RLST"_i32,    ///< For N>=0, vertices [N*3+0, N*3+1, N*3+2] render a screen-aligned rectangle. 0 is upper-left, 1 is upper-right, and 2 is the lower-left corner.
		kPrimitiveTypeLineLoop = "LLOP"_i32,    ///< Like <c>kPrimitiveTypeLineStrip</c>, but the first and last vertices also render a line.
		kPrimitiveTypeQuadList = "QLST"_i32,    ///< For N>=0, vertices [N*4+0, N*4+1, N*4+2] and [N*4+0, N*4+2, N*4+3] render triangles.
		kPrimitiveTypeQuadStrip = "QSTR"_i32,   ///< For N>=0, vertices [N*2+0, N*2+1, N*2+3] and [N*2+0, N*2+3, N*2+2] render triangles.
		kPrimitiveTypePolygon = "POLY"_i32,     ///< For N>=0, vertices [0, N+1, N+2] render a triangle.
	};

    std::ostream& operator<<(std::ostream& out, PrimitiveType type);
  
    class SceneObjectMesh : public BaseSceneObject
    {
        protected:
            std::vector<SceneObjectIndexArray>  m_IndexArray;
            std::vector<SceneObjectVertexArray> m_VertexArray;
			PrimitiveType	m_PrimitiveType;

            bool        m_bVisible;
            bool        m_bShadow;
            bool        m_bMotionBlur;
            
        public:
            SceneObjectMesh(bool visible = true, bool shadow = true, bool motion_blur = true) : BaseSceneObject(SceneObjectType::kSceneObjectTypeMesh), m_bVisible(visible), m_bShadow(shadow), m_bMotionBlur(motion_blur) {};
            SceneObjectMesh(SceneObjectMesh&& mesh)
                : BaseSceneObject(SceneObjectType::kSceneObjectTypeMesh), 
                m_IndexArray(std::move(mesh.m_IndexArray)),
                m_VertexArray(std::move(mesh.m_VertexArray)),
                m_PrimitiveType(mesh.m_PrimitiveType),
                m_bVisible(mesh.m_bVisible),
                m_bShadow(mesh.m_bShadow),
                m_bMotionBlur(mesh.m_bMotionBlur)
            {
            };
            void AddIndexArray(SceneObjectIndexArray&& array) { m_IndexArray.push_back(std::move(array)); };
            void AddVertexArray(SceneObjectVertexArray&& array) { m_VertexArray.push_back(std::move(array)); };
			void SetPrimitiveType(PrimitiveType type) { m_PrimitiveType = type;  };

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectMesh& obj);
    };

    class SceneObjectTexture : public BaseSceneObject
    {
        protected:
            uint32_t m_nTexCoordIndex;
            std::string m_Name;
            std::shared_ptr<Image> m_pImage;
            std::vector<Matrix4X4f> m_Transforms;

        public:
            SceneObjectTexture() : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_nTexCoordIndex(0) {};
            SceneObjectTexture(std::string& name) : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_nTexCoordIndex(0), m_Name(name) {};
            SceneObjectTexture(uint32_t coord_index, std::shared_ptr<Image>& image) : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_nTexCoordIndex(coord_index), m_pImage(image) {};
            SceneObjectTexture(uint32_t coord_index, std::shared_ptr<Image>&& image) : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_nTexCoordIndex(coord_index), m_pImage(std::move(image)) {};
            SceneObjectTexture(SceneObjectTexture&) = default;
            SceneObjectTexture(SceneObjectTexture&&) = default;
            void SetName(std::string& name) { m_Name = name; };
            void AddTransform(Matrix4X4f& matrix) { m_Transforms.push_back(matrix); };

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectTexture& obj);
    };

    template <typename T>
    struct ParameterValueMap
    {
        T Value;
        std::shared_ptr<SceneObjectTexture> ValueMap;

        ParameterValueMap() = default;

        ParameterValueMap(const T value) : Value(value) {};
        ParameterValueMap(const std::shared_ptr<SceneObjectTexture>& value) : ValueMap(value) {};

        ParameterValueMap(const ParameterValueMap<T>& rhs) = default;

        ParameterValueMap(ParameterValueMap<T>&& rhs) = default;

        ParameterValueMap& operator=(const ParameterValueMap<T>& rhs) = default;
        ParameterValueMap& operator=(ParameterValueMap<T>&& rhs) = default;
        ParameterValueMap& operator=(const std::shared_ptr<SceneObjectTexture>& rhs) 
        {
            ValueMap = rhs;
            return *this;
        };

        ~ParameterValueMap() = default;

        friend std::ostream& operator<<(std::ostream& out, const ParameterValueMap<T>& obj)
        {
            out << "Parameter Value: " << obj.Value << std::endl;
            if (obj.ValueMap) {
                out << "Parameter Map: " << *obj.ValueMap << std::endl;
            }

            return out;
        }
    };

    typedef ParameterValueMap<Vector4f> Color;
    typedef ParameterValueMap<Vector3f> Normal;
    typedef ParameterValueMap<float>    Parameter;

    class SceneObjectMaterial : public BaseSceneObject
    {
        protected:
            std::string m_Name;
            Color       m_BaseColor;
            Parameter   m_Metallic;
            Parameter   m_Roughness;
            Normal      m_Normal;
            Parameter   m_Specular;
            Parameter   m_AmbientOcclusion;

        public:
            SceneObjectMaterial(const std::string& name) : BaseSceneObject(SceneObjectType::kSceneObjectTypeMaterial), m_Name(name) {};
            SceneObjectMaterial(std::string&& name) : BaseSceneObject(SceneObjectType::kSceneObjectTypeMaterial), m_Name(std::move(name)) {};
            SceneObjectMaterial(const std::string& name = "", Color&& base_color = Vector4f(1.0f), Parameter&& metallic = 0.0f, Parameter&& roughness = 0.0f, Normal&& normal = Vector3f(0.0f, 0.0f, 1.0f), Parameter&& specular = 0.0f, Parameter&& ao = 0.0f) : BaseSceneObject(SceneObjectType::kSceneObjectTypeMaterial), m_Name(name), m_BaseColor(std::move(base_color)), m_Metallic(std::move(metallic)), m_Roughness(std::move(roughness)), m_Normal(std::move(normal)), m_Specular(std::move(specular)), m_AmbientOcclusion(std::move(ao)) {};
            void SetName(const std::string& name) { m_Name = name; };
            void SetName(std::string&& name) { m_Name = std::move(name); };
            void SetColor(std::string& attrib, Vector4f& color) 
            { 
                if(attrib == "deffuse") {
                    m_BaseColor = Color(color); 
                }
            };

            void SetParam(std::string& attrib, float param) 
            { 
            };

            void SetTexture(std::string& attrib, std::string& textureName) 
            { 
                if(attrib == "diffuse") {
                    m_BaseColor = std::make_shared<SceneObjectTexture>(textureName); 
                }
            };

            void SetTexture(std::string& attrib, std::shared_ptr<SceneObjectTexture>& texture) 
            { 
                if(attrib == "diffuse") {
                    m_BaseColor = texture; 
                }
            };

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectMaterial& obj);
    };

    class SceneObjectGeometry : public BaseSceneObject
    {
        protected:
            std::vector<std::shared_ptr<SceneObjectMesh>> m_Mesh;
			bool        m_bVisible;
			bool        m_bShadow;
			bool        m_bMotionBlur;

        public:
            SceneObjectGeometry() : BaseSceneObject(SceneObjectType::kSceneObjectTypeGeometry) {};

			void SetVisibility(bool visible) { m_bVisible = visible; };
			const bool Visible() { return m_bVisible; };
			void SetIfCastShadow(bool shadow) { m_bShadow = shadow; };
			const bool CastShadow() { return m_bShadow; };
			void SetIfMotionBlur(bool motion_blur) { m_bMotionBlur = motion_blur; };
			const bool MotionBlur() { return m_bMotionBlur; };

            void AddMesh(std::shared_ptr<SceneObjectMesh>& mesh) { m_Mesh.push_back(std::move(mesh)); };

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectGeometry& obj);
    };

    typedef float (*AttenFunc)(float /* Intensity */, float /* Distance */);

    float DefaultAttenFunc(float intensity, float distance);

    class SceneObjectLight : public BaseSceneObject
    {
        protected:
            Color       m_LightColor;
            float       m_fIntensity;
            AttenFunc   m_LightAttenuation;
            float       m_fNearClipDistance;
            float       m_fFarClipDistance;
            bool        m_bCastShadows;

        protected:
            // can only be used as base class of delivered lighting objects
            SceneObjectLight(Color&& color = Vector4f(1.0f), float intensity = 10.0f, AttenFunc atten_func = DefaultAttenFunc, float near_clip = 1.0f, float far_clip = 100.0f, bool cast_shadows = false) : BaseSceneObject(SceneObjectType::kSceneObjectTypeLight), m_LightColor(std::move(color)), m_fIntensity(intensity), m_LightAttenuation(atten_func), m_fNearClipDistance(near_clip), m_fFarClipDistance(far_clip), m_bCastShadows(cast_shadows) {};

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectLight& obj);
    };

    class SceneObjectOmniLight : public SceneObjectLight
    {
        public:
            using SceneObjectLight::SceneObjectLight;

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectOmniLight& obj);
    };

    class SceneObjectSpotLight : public SceneObjectLight
    {
        protected:
            float   m_fConeAngle;
            float   m_fPenumbraAngle;
        public:
            SceneObjectSpotLight(Color&& color = Vector4f(1.0f), float intensity = 10.0f, AttenFunc atten_func = DefaultAttenFunc, float near_clip = 1.0f, float far_clip = 100.0f, bool cast_shadows = false, float cone_angle = PI / 4.0f, float penumbra_angle = PI / 3.0f) : SceneObjectLight(std::move(color), intensity, atten_func, near_clip, far_clip, cast_shadows), m_fConeAngle(cone_angle), m_fPenumbraAngle(penumbra_angle) {};

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectSpotLight& obj);
    };

    class SceneObjectCamera : public BaseSceneObject
    {
        protected:
            float m_fAspect;
            float m_fNearClipDistance;
            float m_fFarClipDistance;

        protected:
            // can only be used as base class
            SceneObjectCamera(float aspect = 16.0f / 9.0f, float near_clip = 1.0f, float far_clip = 100.0f) : BaseSceneObject(SceneObjectType::kSceneObjectTypeCamera), m_fAspect(aspect), m_fNearClipDistance(near_clip), m_fFarClipDistance(far_clip) {};

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectCamera& obj);
    };

    class SceneObjectOrthogonalCamera : public SceneObjectCamera
    {
        public:
            using SceneObjectCamera::SceneObjectCamera;

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectOrthogonalCamera& obj);
    };

    class SceneObjectPerspectiveCamera : public SceneObjectCamera
    {
        protected:
            float m_fFov;

        public:
            SceneObjectPerspectiveCamera(float aspect = 16.0f / 9.0f, float near_clip = 1.0f, float far_clip = 100.0f, float fov = PI / 2.0) : SceneObjectCamera(aspect, near_clip, far_clip), m_fFov(fov) {};

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectPerspectiveCamera& obj);
    };

    class SceneObjectTransform
    {
        protected:
            Matrix4X4f m_matrix;
            bool m_bSceneObjectOnly;

        public:
            SceneObjectTransform() { BuildIdentityMatrix(m_matrix); m_bSceneObjectOnly = false; };

            SceneObjectTransform(const Matrix4X4f& matrix, const bool object_only = false) { m_matrix = matrix; m_bSceneObjectOnly = object_only; };

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectTransform& obj);
    };

    class SceneObjectTranslation : public SceneObjectTransform
    {
        public:
            SceneObjectTranslation(const char axis, const float amount)  
            { 
                switch (axis) {
                    case 'x':
                        MatrixTranslation(m_matrix, amount, 0.0f, 0.0f);
                        break;
                    case 'y':
                        MatrixTranslation(m_matrix, 0.0f, amount, 0.0f);
                        break;
                    case 'z':
                        MatrixTranslation(m_matrix, 0.0f, 0.0f, amount);
                        break;
                    default:
                        assert(0);
                }
            }

            SceneObjectTranslation(const float x, const float y, const float z) 
            {
                MatrixTranslation(m_matrix, x, y, z);
            }
    };

    class SceneObjectRotation : public SceneObjectTransform
    {
        public:
            SceneObjectRotation(const char axis, const float theta)
            {
                switch (axis) {
                    case 'x':
                        MatrixRotationX(m_matrix, theta);
                        break;
                    case 'y':
                        MatrixRotationY(m_matrix, theta);
                        break;
                    case 'z':
                        MatrixRotationZ(m_matrix, theta);
                        break;
                    default:
                        assert(0);
                }
            }

            SceneObjectRotation(Vector3f& axis, const float theta)
            {
                Normalize(axis);
                MatrixRotationAxis(m_matrix, axis, theta);
            }

            SceneObjectRotation(const Quaternion quaternion)
            {
                MatrixRotationQuaternion(m_matrix, quaternion);
            }
    };

    class SceneObjectScale : public SceneObjectTransform
    {
        public:
            SceneObjectScale(const char axis, const float amount)  
            { 
                switch (axis) {
                    case 'x':
                        MatrixScale(m_matrix, amount, 0.0f, 0.0f);
                        break;
                    case 'y':
                        MatrixScale(m_matrix, 0.0f, amount, 0.0f);
                        break;
                    case 'z':
                        MatrixScale(m_matrix, 0.0f, 0.0f, amount);
                        break;
                    default:
                        assert(0);
                }
            }

            SceneObjectScale(const float x, const float y, const float z) 
            {
                MatrixScale(m_matrix, x, y, z);
            }
    };
}

