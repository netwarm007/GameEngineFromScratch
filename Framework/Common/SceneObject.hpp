#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include "Guid.hpp"
#include "Image.hpp"
#include "portable.hpp"

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
    };

    std::ostream& operator<<(std::ostream& out, SceneObjectType type)
    {
        int32_t n = static_cast<int32_t>(type);
        n = endian_net_unsigned_int<int32_t>(n);
        char* c = reinterpret_cast<char*>(&n);
         
        for (int i = 0; i < sizeof(int32_t); i++) {
            out << *c++;
        }

        return out;
    }

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
            
        private:
            // a type must be specified
            BaseSceneObject() = delete; 
            // can not be copied
            BaseSceneObject(BaseSceneObject& obj) = delete;
            BaseSceneObject& operator=(BaseSceneObject& obj) = delete;

        public:
            const Guid& GetGuid() const { return m_Guid; };

        friend std::ostream& operator<<(std::ostream& out, const BaseSceneObject& obj)
        {
            out << "SceneObject" << std::endl;
            out << "-----------" << std::endl;
            out << "GUID: " << obj.m_Guid << std::endl;
            out << "Type: " << obj.m_Type << std::endl;

            return out;
        }
    };

    ENUM(VertexDataType) {
        kVertexDataTypeFloat    = "FLOT"_i32,
        kVertexDataTypeDouble   = "DOUB"_i32
    };

    class SceneObjectVertexArray : public BaseSceneObject
    {
        protected:
            std::string m_Attribute;
            uint32_t    m_MorphTargetIndex;
            VertexDataType m_DataType;

            union {
                float*      m_pDataFloat;
                double*     m_pDataDouble;
            };
            size_t      m_szData;
    };

    ENUM(IndexDataType) {
        kIndexDataTypeInt16 = "_I16"_i32,
        kIndexDataTypeInt32 = "_I32"_i32,
    };

    class SceneObjectIndexArray : public BaseSceneObject
    {
        protected:
            uint32_t    m_MaterialIndex;
            size_t      m_RestartIndex;
            IndexDataType m_DataType;

            union {
                uint16_t*   m_pDataI16;
                uint32_t*   m_pDataI32;
            };
    };

    class SceneObjectMesh : public BaseSceneObject
    {
        protected:
            std::vector<SceneObjectIndexArray>  m_IndexArray;
            std::vector<SceneObjectVertexArray> m_VertexArray;

            bool        m_bVisible      = true;
            bool        m_bShadow       = false;
            bool        m_bMotionBlur   = false;
            
        public:
            SceneObjectMesh() : BaseSceneObject(SceneObjectType::kSceneObjectTypeMesh) {};
    };

    template <typename T>
    struct ParameterMap
    {
        bool bUsingSingleValue = true;

        union _ParameterMap {
            T Value;
            std::shared_ptr<Image> Map;
        };
    };

    typedef ParameterMap<Vector4f> Color;
    typedef ParameterMap<Vector3f> Normal;
    typedef ParameterMap<float>    Parameter;

    class SceneObjectMaterial : public BaseSceneObject
    {
        protected:
            Color       m_BaseColor;
            Parameter   m_Metallic;
            Parameter   m_Roughness;
            Normal      m_Normal;
            Parameter   m_Specular;
            Parameter   m_AmbientOcclusion;

        public:
            SceneObjectMaterial() : BaseSceneObject(SceneObjectType::kSceneObjectTypeMaterial) {};
    };

    typedef float (*AttenFunc)(float /* Intensity */, float /* Distance */);

    class SceneObjectLight : public BaseSceneObject
    {
        protected:
            Color       m_LightColor;
            float       m_Intensity;
            AttenFunc   m_LightAttenuation;
            float       m_fNearClipDistance;
            float       m_fFarClipDistance;
            bool        m_bCastShadows;

        protected:
            // can only be used as base class of delivered lighting objects
            SceneObjectLight() : BaseSceneObject(SceneObjectType::kSceneObjectTypeLight) {};
    };

    class SceneObjectOmniLight : public SceneObjectLight
    {
        public:
            using SceneObjectLight::SceneObjectLight;
    };

    class SceneObjectSpotLight : public SceneObjectLight
    {
        protected:
            float   m_fConeAngle;
            float   m_fPenumbraAngle;
        public:
            using SceneObjectLight::SceneObjectLight;
    };

    class SceneObjectCamera : public BaseSceneObject
    {
        protected:
            float m_fFov;
            float m_fNearClipDistance;
            float m_fFarClipDistance;
        public:
            SceneObjectCamera() : BaseSceneObject(SceneObjectType::kSceneObjectTypeCamera) {};
    };
}

