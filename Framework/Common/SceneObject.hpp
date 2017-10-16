#pragma once
#include <iostream>
#include "Guid.hpp"
#include "Mesh.h"

namespace My {
    namespace details {
        constexpr int32_t i32(const char* s, int32_t v) {
            return *s ? i32(s+1, v * 256 + *s) : v;
        }
    }

    constexpr int32_t operator "" _i32(const char* s, size_t) {
        return details::i32(s, 0);
    }

    enum class SceneObjectType : int32_t {
        kSceneObjectTypeMesh    =   "MESH"_i32,
        kSceneObjectTypeMaterial=   "MATL"_i32,
        kSceneObjectTypeLight   =   "LGHT"_i32,
        kSceneObjectTypeCamera  =   "CAMR"_i32,
        kSceneObjectTypeAnimator=   "ANIM"_i32,
        kSceneObjectTypeClip    =   "CLIP"_i32,
    };

    std::ostream& operator<<(std::ostream& out, SceneObjectType type)
    {
        char* c = reinterpret_cast<char*>(&type);
         
        for (int32_t i = 0; i < sizeof(int32_t); i++) {
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
            BaseSceneObject(BaseSceneObject&& obj) : m_Guid(std::move(obj.m_Guid)), m_Type(obj.m_Type) {};
            BaseSceneObject& operator=(BaseSceneObject&& obj) { this->m_Guid = std::move(obj.m_Guid); this->m_Type = obj.m_Type; return *this; };
            
        private:
            // a type must be specified
            BaseSceneObject() = delete; 
            // can not be copied
            BaseSceneObject(BaseSceneObject& obj) = delete;
            BaseSceneObject& operator=(BaseSceneObject& obj) = delete;

        public:
            const std::string& GetGuid() const { return m_Guid; };

        friend std::ostream& operator<<(std::ostream& out, const BaseSceneObject& obj)
        {
            out << "SceneObject" << std::endl;
            out << "-----------" << std::endl;
            out << obj.m_Guid        << std::endl;
            out << obj.m_Type        << std::endl;

            return out;
        }
    };

    class SceneObjectMesh : public BaseSceneObject
    {
        protected:
            SimpleMesh m_mesh;
            
        public:
            SceneObjectMesh() : BaseSceneObject(SceneObjectType::kSceneObjectTypeMesh) {};
    };
}

