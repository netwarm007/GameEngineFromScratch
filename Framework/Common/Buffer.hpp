#pragma once
#include <memory>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace My {
    class Buffer {
    public:
        Buffer() : m_pData(nullptr), m_szSize(0) {}

        Buffer(size_t size, size_t alignment = 4) : m_szSize(size) { m_pData = reinterpret_cast<uint8_t*>(new uint8_t[size]); }

        Buffer(const Buffer& rhs) { 
            m_pData = reinterpret_cast<uint8_t*>(new uint8_t[rhs.m_szSize]); 
            memcpy(m_pData, rhs.m_pData, rhs.m_szSize);
            m_szSize =  rhs.m_szSize;
        }

        Buffer(Buffer&& rhs) {
            m_pData = rhs.m_pData;
            m_szSize = rhs.m_szSize;
            rhs.m_pData = nullptr;
            rhs.m_szSize = 0;
        }

        Buffer& operator = (const Buffer& rhs) { 
            if (m_szSize >= rhs.m_szSize) {
                memcpy(m_pData, rhs.m_pData, rhs.m_szSize);
            } 
            else {
                if (m_pData) delete[] m_pData; 
                m_pData = reinterpret_cast<uint8_t*>(new uint8_t[rhs.m_szSize]); 
                memcpy(m_pData, rhs.m_pData, rhs.m_szSize);
                m_szSize =  rhs.m_szSize;
            }
            return *this; 
        }

        Buffer& operator = (Buffer&& rhs) { 
            if (m_pData) delete[] m_pData; 
            m_pData = rhs.m_pData;
            m_szSize = rhs.m_szSize;
            rhs.m_pData = nullptr;
            rhs.m_szSize = 0;
            return *this; 
        }

        ~Buffer() { if (m_pData) delete[] m_pData; m_pData = nullptr; }

        uint8_t* GetData() { return m_pData; };
        const uint8_t* GetData() const { return m_pData; };
        size_t GetDataSize() const { return m_szSize; };
        uint8_t* MoveData() 
        { 
            uint8_t* tmp = m_pData;
            m_pData = nullptr;
            m_szSize = 0;
            return tmp;
        }

    protected:
        uint8_t* m_pData;
        size_t m_szSize;
    };
}

