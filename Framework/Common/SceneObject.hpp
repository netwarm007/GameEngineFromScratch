#pragma once
#include "Guid.hpp"

namespace My {
    namespace details {
        constexpr int32 i32(const char* s, int32 v) {
            return *s ? i32(s+1, v * 256 + *s) : v;
        }
    }

    constexpr int32 operator "" _i32(const char* s, unsigned long) {
        return details::i32(s, 0);
    }

    enum class SceneObjectType : int32_t {
        kSceneObjectTypeMesh    =   "MESH"_i32,
        kSceneObjectTypeMaterial=   "MATL"_i32,
        kSceneObjectTypeLight   =   "LGHT"_i32,
        kSceneObjectTypeCamera  =   "CAMR"_i32,
        kSceneObjectTypeAnimator=   "ANIM"_i32,
        kSceneObjectTypeClip    =   "CLIP"_i32,
    }

    class BaseSceneObject
    {
        protected:
            Guid m_Guid;
            SceneObjectType m_Type;
        protected:
            // can only be used as base class
            BaseSceneObject(SceneObjectType type) : m_Type(type) {};
            BaseSceneObject(BaseSceneObject&& obj) : m_Guid(std::move(obj.m_Guid)), m_Type(obj.type) {};
            operator=(BaseSceneObject&& obj) { this->m_Guid = std::move(obj.m_Guid; this->m_Type = obj.m_Type; };
            
        private:
            // a type must be specified
            BaseSceneObject() = deleted; 
            // can not be copied
            BaseSceneObject(BaseSceneObject& obj) = deleted;
            operator=(BaseSceneObject& obj) = deleted;
        public:
            const Guid& GetGuid() const { return m_Guid; };
    };
}

