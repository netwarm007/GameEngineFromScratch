#pragma once
#include <vector>

#include "portable.hpp"

namespace My {
ENUM(CurveType){kLinear = "LINE"_i32, kBezier = "BEZI"_i32};

class CurveBase {
   private:
    const CurveType m_kCurveType;

   public:
    CurveBase() = delete;
    explicit CurveBase(CurveType type) : m_kCurveType(type) {}
    virtual ~CurveBase() = default;
    [[nodiscard]] CurveType GetCurveType() const { return m_kCurveType; }
};

template <typename TVAL, typename TPARAM>
class Curve {
   protected:
    std::vector<TVAL> m_Knots;

   public:
    virtual TPARAM Reverse(TVAL t, size_t& index) const = 0;
    [[nodiscard]] virtual TVAL Interpolate(TPARAM t,
                                           const size_t index) const = 0;
    void AddKnot(const TVAL knot) { m_Knots.push_back(knot); }
};
}  // namespace My
