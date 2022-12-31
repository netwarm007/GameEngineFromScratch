#pragma once
#include "Ability.hpp"
#include "portable.hpp"

namespace My {
template <class MaterialPtr>
Ability MaterialContainer {
   protected:
    __device__ virtual ~MaterialContainer() {
#ifdef __CUDACC__
        if (m_ptrMat) delete m_ptrMat;
#endif
    }

   public:
    using MaterialPtrType = MaterialPtr;

    __device__ void SetMaterial(MaterialPtr m) {
        m_ptrMat = m;
    }

    __device__ MaterialPtr GetMaterial() {
        return m_ptrMat;
    }

   protected:
    MaterialPtr m_ptrMat;
};
}  // namespace My
