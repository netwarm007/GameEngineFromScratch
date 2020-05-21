#pragma once
#include <string>

#include "SceneObjectTypeDef.hpp"

namespace My {
enum VertexAttribute : unsigned long {
    VertexAttributePosition,
    VertexAttributeNormal,
    VertexAttributeTexcoord,
    VertexAttributeTangent
};

class SceneObjectVertexArray {
   protected:
    const std::string m_strAttribute;
    const uint32_t m_nMorphTargetIndex{0};
    const VertexDataType m_DataType{VertexDataType::kVertexDataTypeFloat3};

    const uint8_t* m_pData;

    const size_t m_szData;

   public:
    explicit SceneObjectVertexArray(
        const char* attr = "", const uint32_t morph_index = 0,
        const VertexDataType data_type = VertexDataType::kVertexDataTypeFloat3,
        const uint8_t* data = nullptr, const size_t data_size = 0)
        : m_strAttribute(attr),
          m_nMorphTargetIndex(morph_index),
          m_DataType(data_type),
          m_pData(data),
          m_szData(data_size){};

    SceneObjectVertexArray(const SceneObjectVertexArray& rhs) = delete;

    SceneObjectVertexArray(SceneObjectVertexArray&& rhs) noexcept
        : m_strAttribute(std::move(rhs.m_strAttribute)),
          m_nMorphTargetIndex(rhs.m_nMorphTargetIndex),
          m_DataType(rhs.m_DataType),
          m_szData(rhs.m_szData) {
        m_pData = rhs.m_pData;
        rhs.m_pData = nullptr;
    }

    ~SceneObjectVertexArray() {
        if (m_pData) delete[] m_pData;
    }

    [[nodiscard]] const std::string& GetAttributeName() const {
        return m_strAttribute;
    };
    [[nodiscard]] VertexDataType GetDataType() const { return m_DataType; };
    [[nodiscard]] size_t GetDataSize() const {
        size_t size = m_szData;

        switch (m_DataType) {
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
    [[nodiscard]] const void* GetData() const { return m_pData; };
    [[nodiscard]] size_t GetVertexCount() const {
        size_t size = m_szData;

        switch (m_DataType) {
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

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectVertexArray& obj);
};
}  // namespace My
