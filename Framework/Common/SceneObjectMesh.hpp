#pragma once
#include "BaseSceneObject.hpp"
#include "ConvexHull.hpp"
#include "SceneObjectIndexArray.hpp"
#include "SceneObjectTypeDef.hpp"
#include "SceneObjectVertexArray.hpp"
#include "geommath.hpp"
#include <vector>

namespace My {
    class SceneObjectMesh : public BaseSceneObject
    {
        protected:
            std::vector<SceneObjectIndexArray>  m_IndexArray;
            std::vector<SceneObjectVertexArray> m_VertexArray;
			PrimitiveType	m_PrimitiveType;

        public:
            SceneObjectMesh(bool visible = true, bool shadow = true, bool motion_blur = true) : BaseSceneObject(SceneObjectType::kSceneObjectTypeMesh) {};
            SceneObjectMesh(SceneObjectMesh&& mesh) noexcept
                : BaseSceneObject(SceneObjectType::kSceneObjectTypeMesh), 
                m_IndexArray(std::move(mesh.m_IndexArray)),
                m_VertexArray(std::move(mesh.m_VertexArray)),
                m_PrimitiveType(mesh.m_PrimitiveType)
            {
            };
            void AddIndexArray(SceneObjectIndexArray&& array) { m_IndexArray.push_back(std::move(array)); };
            void AddVertexArray(SceneObjectVertexArray&& array) { m_VertexArray.push_back(std::move(array)); };
			void SetPrimitiveType(PrimitiveType type) { m_PrimitiveType = type;  };

            [[nodiscard]] size_t GetIndexGroupCount() const { return m_IndexArray.size(); };
            [[nodiscard]] size_t GetIndexCount(const size_t index) const { return (m_IndexArray.empty()? 0 : m_IndexArray[index].GetIndexCount()); };
            [[nodiscard]] size_t GetVertexCount() const { return (m_VertexArray.empty()? 0 : m_VertexArray[0].GetVertexCount()); };
            [[nodiscard]] uint32_t GetVertexPropertiesCount() const { return static_cast<uint32_t>(m_VertexArray.size()); }; 
            [[nodiscard]] const SceneObjectVertexArray& GetVertexPropertyArray(const size_t index) const { return m_VertexArray[index]; };
            [[nodiscard]] const SceneObjectIndexArray& GetIndexArray(const size_t index) const { return m_IndexArray[index]; };
            const PrimitiveType& GetPrimitiveType() { return m_PrimitiveType; };
            [[nodiscard]] BoundingBox GetBoundingBox() const;
            [[nodiscard]] ConvexHull GetConvexHull() const;

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectMesh& obj);
    };
}
