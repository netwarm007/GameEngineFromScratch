#pragma once
#include "SceneObjectTypeDef.hpp"

namespace My {
class SceneObjectIndexArray {
   protected:
    const uint32_t m_nMaterialIndex;
    const size_t m_szRestartIndex;
    const IndexDataType m_DataType;

    const uint8_t* m_pData;

    const size_t m_szData;

   public:
    explicit SceneObjectIndexArray(
        const uint32_t material_index = 0, const size_t restart_index = 0,
        const IndexDataType data_type = IndexDataType::kIndexDataTypeInt16,
        const uint8_t* data = nullptr, const size_t data_size = 0)
        : m_nMaterialIndex(material_index),
          m_szRestartIndex(restart_index),
          m_DataType(data_type),
          m_pData(data),
          m_szData(data_size){};

    SceneObjectIndexArray(const SceneObjectIndexArray& rhs) = delete;

    SceneObjectIndexArray(SceneObjectIndexArray&& rhs) noexcept
        : m_nMaterialIndex(rhs.m_nMaterialIndex),
          m_szRestartIndex(rhs.m_szRestartIndex),
          m_DataType(rhs.m_DataType),
          m_szData(rhs.m_szData) {
        m_pData = rhs.m_pData;
        rhs.m_pData = nullptr;
    }

    ~SceneObjectIndexArray() {
        if (m_pData) delete[] m_pData;
    }

    [[nodiscard]] uint32_t GetMaterialIndex() const {
        return m_nMaterialIndex;
    };
    [[nodiscard]] IndexDataType GetIndexType() const { return m_DataType; };
    [[nodiscard]] const void* GetData() const { return m_pData; };
    [[nodiscard]] size_t GetDataSize() const {
        size_t size = m_szData;

        switch (m_DataType) {
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

    [[nodiscard]] size_t GetIndexCount() const { return m_szData; }

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectIndexArray& obj);
};
}  // namespace My