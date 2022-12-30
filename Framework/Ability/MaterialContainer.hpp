#pragma once
#include "Ability.hpp"

namespace My {
template <class MaterialPtr>
Ability MaterialContainer {
   public:
    __device__ virtual ~MaterialContainer() {}
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
