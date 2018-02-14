#pragma once
#include <assert.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "Guid.hpp"
#include "Image.hpp"
#include "portable.hpp"
#include "geommath.hpp"
#include "AssetLoader.hpp"
#include "JPEG.hpp"
#include "PNG.hpp"
#include "BMP.hpp"
#include "TGA.hpp"

namespace My {
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

    ENUM(SceneObjectCollisionType) {
        kSceneObjectCollisionTypeNone   =   "CNON"_i32,
        kSceneObjectCollisionTypeSphere =   "CSPH"_i32,
        kSceneObjectCollisionTypeBox    =   "CBOX"_i32,
        kSceneObjectCollisionTypeCylinder = "CCYL"_i32,
        kSceneObjectCollisionTypeCapsule  = "CCAP"_i32,
        kSceneObjectCollisionTypeCone   =   "CCON"_i32,
        kSceneObjectCollisionTypeMultiSphere = "CMUL"_i32,
        kSceneObjectCollisionTypeConvexHull =  "CCVH"_i32,
        kSceneObjectCollisionTypeConvexMesh =  "CCVM"_i32,
        kSceneObjectCollisionTypeBvhMesh =  "CBVM"_i32,
        kSceneObjectCollisionTypeHeightfield = "CHIG"_i32,
        kSceneObjectCollisionTypePlane  =   "CPLN"_i32,
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
        kVertexDataTypeDouble4   = "DUB4"_i32
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

            const std::string& GetAttributeName() const { return m_strAttribute; };
            VertexDataType GetDataType() const { return m_DataType; };
            size_t GetDataSize() const 
            { 
                size_t size = m_szData;

                switch(m_DataType) {
                    case VertexDataType::kVertexDataTypeFloat1:
                    case VertexDataType::kVertexDataTypeFloat2:
                    case VertexDataType::kVertexDataTypeFloat3:
                    case VertexDataType::kVertexDataTypeFloat4:
                        size *= sizeof(float);
                        break;
                    case VertexDataType::kVertexDataTypeDouble1:
                    case VertexDataType::kVertexDataTypeDouble2:
                    case VertexDataType::kVertexDataTypeDouble3:
                    case VertexDataType::kVertexDataTypeDouble4:
                        size *= sizeof(double);
                        break;
                    default:
                        size = 0;
                        assert(0);
                        break;
                }

                return size;
            }; 
            const void* GetData() const { return m_pData; };
            size_t GetVertexCount() const
            {
                size_t size = m_szData;

                switch(m_DataType) {
                    case VertexDataType::kVertexDataTypeFloat1:
                        size /= 1;
                        break;
                    case VertexDataType::kVertexDataTypeFloat2:
                        size /= 2;
                        break;
                    case VertexDataType::kVertexDataTypeFloat3:
                        size /= 3;
                        break;
                    case VertexDataType::kVertexDataTypeFloat4:
                        size /= 4;
                        break;
                    case VertexDataType::kVertexDataTypeDouble1:
                        size /= 1;
                        break;
                    case VertexDataType::kVertexDataTypeDouble2:
                        size /= 2;
                        break;
                    case VertexDataType::kVertexDataTypeDouble3:
                        size /= 3;
                        break;
                    case VertexDataType::kVertexDataTypeDouble4:
                        size /= 4;
                        break;
                    default:
                        size = 0;
                        assert(0);
                        break;
                }

                return size;
            }

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

            const uint32_t GetMaterialIndex() const { return m_nMaterialIndex; };
            const IndexDataType GetIndexType() const { return m_DataType; };
            const void* GetData() const { return m_pData; };
            size_t GetDataSize() const 
            { 
                size_t size = m_szData;

                switch(m_DataType) {
                    case IndexDataType::kIndexDataTypeInt8:
                        size *= sizeof(int8_t);
                        break;
                    case IndexDataType::kIndexDataTypeInt16:
                        size *= sizeof(int16_t);
                        break;
                    case IndexDataType::kIndexDataTypeInt32:
                        size *= sizeof(int32_t);
                        break;
                    case IndexDataType::kIndexDataTypeInt64:
                        size *= sizeof(int64_t);
                        break;
                    default:
                        size = 0;
                        assert(0);
                        break;
                }

                return size;
            };

            size_t GetIndexCount() const
            {
                return m_szData;
            }

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

        public:
            SceneObjectMesh(bool visible = true, bool shadow = true, bool motion_blur = true) : BaseSceneObject(SceneObjectType::kSceneObjectTypeMesh) {};
            SceneObjectMesh(SceneObjectMesh&& mesh)
                : BaseSceneObject(SceneObjectType::kSceneObjectTypeMesh), 
                m_IndexArray(std::move(mesh.m_IndexArray)),
                m_VertexArray(std::move(mesh.m_VertexArray)),
                m_PrimitiveType(mesh.m_PrimitiveType)
            {
            };
            void AddIndexArray(SceneObjectIndexArray&& array) { m_IndexArray.push_back(std::move(array)); };
            void AddVertexArray(SceneObjectVertexArray&& array) { m_VertexArray.push_back(std::move(array)); };
			void SetPrimitiveType(PrimitiveType type) { m_PrimitiveType = type;  };

            size_t GetIndexGroupCount() const { return m_IndexArray.size(); };
            size_t GetIndexCount(const size_t index) const { return (m_IndexArray.empty()? 0 : m_IndexArray[index].GetIndexCount()); };
            size_t GetVertexCount() const { return (m_VertexArray.empty()? 0 : m_VertexArray[0].GetVertexCount()); };
            size_t GetVertexPropertiesCount() const { return m_VertexArray.size(); }; 
            const SceneObjectVertexArray& GetVertexPropertyArray(const size_t index) const { return m_VertexArray[index]; };
            const SceneObjectIndexArray& GetIndexArray(const size_t index) const { return m_IndexArray[index]; };
            const PrimitiveType& GetPrimitiveType() { return m_PrimitiveType; };

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectMesh& obj);
    };

    class SceneObjectTexture : public BaseSceneObject
    {
        protected:
            std::string m_Name;
            uint32_t m_nTexCoordIndex;
            std::shared_ptr<Image> m_pImage;
            std::vector<Matrix4X4f> m_Transforms;

        public:
            SceneObjectTexture() : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_nTexCoordIndex(0) {};
            SceneObjectTexture(const std::string& name) : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_Name(name), m_nTexCoordIndex(0) {};
            SceneObjectTexture(uint32_t coord_index, std::shared_ptr<Image>& image) : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_nTexCoordIndex(coord_index), m_pImage(image) {};
            SceneObjectTexture(uint32_t coord_index, std::shared_ptr<Image>&& image) : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_nTexCoordIndex(coord_index), m_pImage(std::move(image)) {};
            SceneObjectTexture(SceneObjectTexture&) = default;
            SceneObjectTexture(SceneObjectTexture&&) = default;
            void AddTransform(Matrix4X4f& matrix) { m_Transforms.push_back(matrix); };
            void SetName(const std::string& name) { m_Name = name; };
            void SetName(std::string&& name) { m_Name = std::move(name); };
            const std::string& GetName() const { return m_Name; };
            void LoadTexture() {
                if (!m_pImage)
                {
                    // we should lookup if the texture has been loaded already to prevent
                    // duplicated load. This could be done in Asset Loader Manager.
                    Buffer buf = g_pAssetLoader->SyncOpenAndReadBinary(m_Name.c_str());
                    std::string ext = m_Name.substr(m_Name.find_last_of("."));
                    if (ext == ".jpg" || ext == ".jpeg")
                    {
                        JfifParser jfif_parser;
                        m_pImage = std::make_shared<Image>(jfif_parser.Parse(buf));
                    }
                    else if (ext == ".png")
                    {
                        PngParser png_parser;
                        m_pImage = std::make_shared<Image>(png_parser.Parse(buf));
                    }
                    else if (ext == ".bmp")
                    {
                        BmpParser bmp_parser;
                        m_pImage = std::make_shared<Image>(bmp_parser.Parse(buf));
                    }
                    else if (ext == ".tga")
                    {
                        TgaParser tga_parser;
                        m_pImage = std::make_shared<Image>(tga_parser.Parse(buf));
                    }
                }
            }

            const Image& GetTextureImage() 
            { 
                if (!m_pImage)
                {
                    LoadTexture();
                }

                return *m_pImage; 
            };

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
            Color       m_Specular;
            Parameter   m_SpecularPower;
            Parameter   m_AmbientOcclusion;
            Color       m_Opacity;
            Color       m_Transparency;
            Color       m_Emission;

        public:
            SceneObjectMaterial(void) 
                : BaseSceneObject(SceneObjectType::kSceneObjectTypeMaterial), 
                m_Name(""), 
                m_BaseColor(Vector4f(1.0f)), 
                m_Metallic(0.0f), 
                m_Roughness(0.0f), 
                m_Normal(Vector3f(0.0f, 0.0f, 1.0f)), 
                m_Specular(0.0f), 
                m_SpecularPower(1.0f), 
                m_AmbientOcclusion(1.0f), 
                m_Opacity(1.0f), 
                m_Transparency(0.0f), 
                m_Emission(0.0f) {};
            SceneObjectMaterial(const char* name) : SceneObjectMaterial()
                { m_Name = name; };
            SceneObjectMaterial(const std::string& name) : SceneObjectMaterial()
                { m_Name = name; };
            SceneObjectMaterial(std::string&& name) : SceneObjectMaterial()
                { m_Name = std::move(name); };

            const std::string& GetName() const { return m_Name; };
            const Color& GetBaseColor() const { return m_BaseColor; };
            const Color& GetSpecularColor() const { return m_Specular; };
            const Parameter& GetSpecularPower() const { return m_SpecularPower; };
            void SetName(const std::string& name) { m_Name = name; };
            void SetName(std::string&& name) { m_Name = std::move(name); };
            void SetColor(const std::string& attrib, const Vector4f& color) 
            { 
                if(attrib == "diffuse") {
                    m_BaseColor = Color(color); 
                }

                if(attrib == "specular") {
                    m_Specular = Color(color); 
                }

                if(attrib == "emission") {
                    m_Emission = Color(color); 
                }

                if(attrib == "opacity") {
                    m_Opacity = Color(color); 
                }

                if(attrib == "transparency") {
                    m_Transparency = Color(color); 
                }
            };

            void SetParam(const std::string& attrib, const float param) 
            { 

                if(attrib == "specular_power") {
                    m_SpecularPower = Parameter(param); 
                }
            };

            void SetTexture(const std::string& attrib, const std::string& textureName) 
            { 
                if(attrib == "diffuse") {
                    m_BaseColor = std::make_shared<SceneObjectTexture>(textureName); 
                }

                if(attrib == "specular") {
                    m_Specular = std::make_shared<SceneObjectTexture>(textureName); 
                }

                if(attrib == "specular_power") {
                    m_SpecularPower = std::make_shared<SceneObjectTexture>(textureName); 
                }

                if(attrib == "emission") {
                    m_Emission = std::make_shared<SceneObjectTexture>(textureName); 
                }

                if(attrib == "opacity") {
                    m_Opacity = std::make_shared<SceneObjectTexture>(textureName); 
                }

                if(attrib == "transparency") {
                    m_Transparency = std::make_shared<SceneObjectTexture>(textureName); 
                }

                if(attrib == "normal") {
                    m_Normal = std::make_shared<SceneObjectTexture>(textureName); 
                }
            };

            void SetTexture(const std::string& attrib, const std::shared_ptr<SceneObjectTexture>& texture) 
            { 
                if(attrib == "diffuse") {
                    m_BaseColor = texture; 
                }

                if(attrib == "specular") {
                    m_Specular = texture;
                }

                if(attrib == "specular_power") {
                    m_SpecularPower = texture;
                }

                if(attrib == "emission") {
                    m_Emission = texture;
                }

                if(attrib == "opacity") {
                    m_Opacity = texture;
                }

                if(attrib == "transparency") {
                    m_Transparency = texture;
                }

                if(attrib == "normal") {
                    m_Normal = texture;
                }
            };

            void LoadTextures()
            {
                if (m_BaseColor.ValueMap) {
                    m_BaseColor.ValueMap->LoadTexture();
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
            SceneObjectCollisionType m_CollisionType;
            float       m_CollisionParameters[10];

        public:
            SceneObjectGeometry(void) : BaseSceneObject(SceneObjectType::kSceneObjectTypeGeometry), m_CollisionType(SceneObjectCollisionType::kSceneObjectCollisionTypeNone) {};

			void SetVisibility(bool visible) { m_bVisible = visible; };
			const bool Visible() { return m_bVisible; };
			void SetIfCastShadow(bool shadow) { m_bShadow = shadow; };
			const bool CastShadow() { return m_bShadow; };
			void SetIfMotionBlur(bool motion_blur) { m_bMotionBlur = motion_blur; };
			const bool MotionBlur() { return m_bMotionBlur; };
            void SetCollisionType(SceneObjectCollisionType collision_type) { m_CollisionType = collision_type; };
            const SceneObjectCollisionType CollisionType() const { return  m_CollisionType; }
            void SetCollisionParameters(const float* param, int32_t count)
            {
                assert(count > 0 && count < 10);
                memcpy(m_CollisionParameters, param, sizeof(float) * count);
            }
            const float* CollisionParameters() const { return m_CollisionParameters; }

            void AddMesh(std::shared_ptr<SceneObjectMesh>& mesh) { m_Mesh.push_back(std::move(mesh)); };
            const std::weak_ptr<SceneObjectMesh> GetMesh() { return (m_Mesh.empty()? nullptr : m_Mesh[0]); };
            const std::weak_ptr<SceneObjectMesh> GetMeshLOD(size_t lod) { return (lod < m_Mesh.size()? m_Mesh[lod] : nullptr); };

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectGeometry& obj);
    };

    typedef std::function<float(float /* Intensity */, float /* Distance */)> AttenFunc;

    float DefaultAttenFunc(float intensity, float distance);

    class SceneObjectLight : public BaseSceneObject
    {
        protected:
            Color       m_LightColor;
            float       m_fIntensity;
            AttenFunc   m_LightAttenuation;
            bool        m_bCastShadows;
            std::string m_strTexture;

        public:
			void SetIfCastShadow(bool shadow) { m_bCastShadows = shadow; };

            void SetColor(std::string& attrib, Vector4f& color) 
            { 
                if(attrib == "light") {
                    m_LightColor = Color(color); 
                }
            };

            void SetParam(std::string& attrib, float param) 
            { 
                if(attrib == "intensity") {
                    m_fIntensity = param; 
                }
            };

            void SetTexture(std::string& attrib, std::string& textureName) 
            { 
                if(attrib == "projection") {
                    m_strTexture = textureName;
                }
            };

            void SetAttenuation(AttenFunc func)
            {
                m_LightAttenuation = func;
            }

            const Color& GetColor() { return m_LightColor; };
            float GetIntensity() { return m_fIntensity; };

        protected:
            // can only be used as base class of delivered lighting objects
            SceneObjectLight(void) : BaseSceneObject(SceneObjectType::kSceneObjectTypeLight), m_LightColor(Vector4f(1.0f)), m_fIntensity(100.0f), m_LightAttenuation(DefaultAttenFunc), m_bCastShadows(false) {};

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
            SceneObjectSpotLight(void) : SceneObjectLight(), m_fConeAngle(PI / 4.0f), m_fPenumbraAngle(PI / 3.0f) {};

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectSpotLight& obj);
    };

    class SceneObjectInfiniteLight : public SceneObjectLight
    {
        public:
            using SceneObjectLight::SceneObjectLight;

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectInfiniteLight& obj);
    };

    class SceneObjectCamera : public BaseSceneObject
    {
        protected:
            float m_fAspect;
            float m_fNearClipDistance;
            float m_fFarClipDistance;

        public:
            void SetColor(std::string& attrib, Vector4f& color) 
            { 
                // TODO: extension
            };

            void SetParam(std::string& attrib, float param) 
            { 
                if(attrib == "near") {
                    m_fNearClipDistance = param; 
                }
                else if(attrib == "far") {
                    m_fFarClipDistance = param; 
                }
            };

            void SetTexture(std::string& attrib, std::string& textureName) 
            { 
                // TODO: extension
            };

            float GetNearClipDistance() const { return m_fNearClipDistance; };
            float GetFarClipDistance() const { return m_fFarClipDistance; };

        protected:
            // can only be used as base class
            SceneObjectCamera(void) : BaseSceneObject(SceneObjectType::kSceneObjectTypeCamera), m_fAspect(16.0f / 9.0f), m_fNearClipDistance(1.0f), m_fFarClipDistance(100.0f) {};

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
            void SetParam(std::string& attrib, float param) 
            { 
                // TODO: handle fovx, fovy
                if(attrib == "fov") {
                    m_fFov = param; 
                }
                SceneObjectCamera::SetParam(attrib, param);
            };

        public:
            SceneObjectPerspectiveCamera(float fov = PI / 2.0) : SceneObjectCamera(), m_fFov(fov) {};
            float GetFov() const { return m_fFov; };

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

            operator Matrix4X4f() { return m_matrix; };
            operator const Matrix4X4f() const { return m_matrix; };

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

