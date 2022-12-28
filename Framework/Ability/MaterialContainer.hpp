#pragma once
#include "Ability.hpp"

namespace My {
template <class MaterialPtr>
Ability MaterialContainer {
   public:
    virtual ~MaterialContainer() = default;
    using MaterialPtrType = MaterialPtr;

    void SetMaterial(MaterialPtr m) {
        m_ptrMat = m;
    }

    auto GetMaterial() {
        return m_ptrMat;
    }

   protected:
    MaterialPtr m_ptrMat;
};
}  // namespace My
