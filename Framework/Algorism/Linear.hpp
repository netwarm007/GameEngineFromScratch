#pragma once
#include "Curve.hpp"
#include "MatrixComposeDecompose.hpp"
#include "geommath.hpp"

namespace My {
template <typename TVAL, typename TPARAM>
class Linear : public CurveBase, public Curve<TVAL, TPARAM> {
   public:
    Linear() : CurveBase(CurveType::kLinear) {}
    explicit Linear(const std::vector<TVAL> knots) : Linear() {
        Curve<TVAL, TPARAM>::m_Knots = knots;
    }

    Linear(const TVAL* knots, const size_t count) : Linear() {
        for (size_t i = 0; i < count; i++) {
            Curve<TVAL, TPARAM>::m_Knots.push_back(knots[i]);
        }
    }

    TPARAM Reverse(TVAL t, size_t& index) const final {
        TVAL t1{0};
        TVAL t2{0};

        if (Curve<TVAL, TPARAM>::m_Knots.size() < 2) return TPARAM(0);

        if (t <= Curve<TVAL, TPARAM>::m_Knots.front()) {
            index = 0;
            return TPARAM(0);
        }

        if (t >= Curve<TVAL, TPARAM>::m_Knots.back()) {
            index = Curve<TVAL, TPARAM>::m_Knots.size();
            return TPARAM(1);
        }

        for (size_t i = 1; i < Curve<TVAL, TPARAM>::m_Knots.size(); i++) {
            if (t >= Curve<TVAL, TPARAM>::m_Knots[i]) continue;

            t1 = Curve<TVAL, TPARAM>::m_Knots[i - 1];
            t2 = Curve<TVAL, TPARAM>::m_Knots[i];
            index = i;
            break;
        }

        return TPARAM((t - t1) / (t2 - t1));
    }

    [[nodiscard]] TVAL Interpolate(TPARAM s, const size_t index) const final {
        if (Curve<TVAL, TPARAM>::m_Knots.empty()) return static_cast<TVAL>(0);

        if (Curve<TVAL, TPARAM>::m_Knots.size() == 1)
            return Curve<TVAL, TPARAM>::m_Knots[0];
        else if (Curve<TVAL, TPARAM>::m_Knots.size() < index + 1)
            return Curve<TVAL, TPARAM>::m_Knots.back();
        else if (index == 0)
            return Curve<TVAL, TPARAM>::m_Knots.front();
        else {
            auto t1 = Curve<TVAL, TPARAM>::m_Knots[index - 1];
            auto t2 = Curve<TVAL, TPARAM>::m_Knots[index];

            return (TPARAM(1) - s) * t1 + s * t2;
        }
    }
};

template <typename T>
class Linear<Quaternion<T>, T> : public CurveBase,
                                 public Curve<Quaternion<T>, T> {
   public:
    Linear() : CurveBase(CurveType::kLinear) {}
    explicit Linear(const std::vector<Quaternion<T>> knots) : Linear() {
        Curve<Quaternion<T>, T>::m_Knots = knots;
    }

    Linear(const Quaternion<T>* knots, const size_t count) : Linear() {
        for (size_t i = 0; i < count; i++) {
            Curve<Quaternion<T>, T>::m_Knots.push_back(knots[i]);
        }
    }

    T Reverse(Quaternion<T> v, size_t& index) const final {
        T result = 0;
        assert(0);

        return result;
    }

    [[nodiscard]] Quaternion<T> Interpolate(T s,
                                            const size_t index) const final {
        Quaternion<T> result{0};
        assert(0);

        return result;
    }
};

template <>
class Linear<Matrix4X4f, float> : public CurveBase,
                                  public Curve<Matrix4X4f, float> {
   public:
    Linear() : CurveBase(CurveType::kLinear) {}
    explicit Linear(const std::vector<Matrix4X4f>& knots) : Linear() {
        Curve<Matrix4X4f, float>::m_Knots = knots;
    }

    Linear(const Matrix4X4f* knots, const size_t count) : Linear() {
        for (size_t i = 0; i < count; i++) {
            Curve<Matrix4X4f, float>::m_Knots.push_back(knots[i]);
        }
    }

    float Reverse(Matrix4X4f v, size_t& index) const final {
        float result = 0.0f;
        assert(0);

        return result;
    }

    [[nodiscard]] Matrix4X4f Interpolate(float s,
                                         const size_t index) const final {
        Matrix4X4f result;
        BuildIdentityMatrix(result);
        if (Curve<Matrix4X4f, float>::m_Knots.empty()) return result;
        if (Curve<Matrix4X4f, float>::m_Knots.size() == 1)
            return Curve<Matrix4X4f, float>::m_Knots[0];
        else if (Curve<Matrix4X4f, float>::m_Knots.size() < index + 1)
            return Curve<Matrix4X4f, float>::m_Knots.back();
        else if (index == 0)
            return Curve<Matrix4X4f, float>::m_Knots.front();
        else {
            auto v1 = Curve<Matrix4X4f, float>::m_Knots[index - 1];
            auto v2 = Curve<Matrix4X4f, float>::m_Knots[index];

            Vector3f translation1, translation2;
            Vector3f scalar1, scalar2;
            Vector3f rotation1, rotation2;

            Matrix4X4fDecompose(v1, rotation1, scalar1, translation1);
            Matrix4X4fDecompose(v2, rotation2, scalar2, translation2);

            // Interpolate tranlation
            Vector3f translation = (1.0f - s) * translation1 + s * translation2;
            // Interpolate scalar
            Vector3f scalar = (1.0f - s) * scalar1 + s * scalar2;
            // Interpolate rotation
            Vector3f rotation = (1.0f - s) * rotation1 + s * rotation2;

            // compose the interpolated matrix
            Matrix4X4fCompose(result, rotation, scalar, translation);
        }

        return result;
    }
};
}  // namespace My
