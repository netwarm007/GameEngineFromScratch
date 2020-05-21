#pragma once
#include "Guid.hpp"
#include "SceneObjectTypeDef.hpp"
#include "portable.hpp"

namespace My {
using namespace xg;
class BaseSceneObject {
   protected:
    Guid m_Guid;
    SceneObjectType m_Type;

   protected:
    // can only be used as base class
    explicit BaseSceneObject(SceneObjectType type) : m_Type(type) {
        m_Guid = newGuid();
    };
    BaseSceneObject(Guid& guid, SceneObjectType type)
        : m_Guid(guid), m_Type(type){};
    BaseSceneObject(Guid&& guid, SceneObjectType type)
        : m_Guid(guid), m_Type(type){};
    BaseSceneObject(BaseSceneObject&& obj) noexcept
        : m_Guid(obj.m_Guid), m_Type(obj.m_Type){};
    BaseSceneObject& operator=(BaseSceneObject&& obj) noexcept {
        this->m_Guid = obj.m_Guid;
        this->m_Type = obj.m_Type;
        return *this;
    };
    virtual ~BaseSceneObject() = default;

   public:
    // a type must be specified
    BaseSceneObject() = delete;
    // can not be copied
    BaseSceneObject(BaseSceneObject& obj) = delete;
    BaseSceneObject& operator=(BaseSceneObject& obj) = delete;

   public:
    [[nodiscard]] const Guid& GetGuid() const { return m_Guid; };
    [[nodiscard]] SceneObjectType GetType() const { return m_Type; };

    friend std::ostream& operator<<(std::ostream& out,
                                    const BaseSceneObject& obj);
};
}  // namespace My
