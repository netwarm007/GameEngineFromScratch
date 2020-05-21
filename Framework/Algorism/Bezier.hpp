#pragma once
#include <cassert>
#include <map>

#include "Curve.hpp"
#include "geommath.hpp"
#include "numerical.hpp"

namespace My {
template <typename TVAL, typename TPARAM>
class Bezier : public CurveBase, public Curve<TVAL, TPARAM> {
   private:
    std::map<TVAL, TVAL> m_IncomingControlPoints;
    std::map<TVAL, TVAL> m_OutgoingControlPoints;

   public:
    Bezier() : CurveBase(CurveType::kBezier) {}
    Bezier(const std::vector<TVAL> knots, const std::vector<TVAL> incoming_cp,
           const std::vector<TVAL> outgoing_cp)
        : Bezier() {
        assert(knots.size() == incoming_cp.size() &&
               knots.size() == outgoing_cp.size());

        Curve<TVAL, TPARAM>::m_Knots = knots;

        auto count = knots.size();

        for (size_t i = 0; i < count; i++) {
            auto knot = knots[i];
            auto in_cp = incoming_cp[i];
            auto out_cp = outgoing_cp[i];
            AddControlPoints(knot, in_cp, out_cp);
        }
    }

    Bezier(const TVAL* knots, const TVAL* incoming_cp, const TVAL* outgoing_cp,
           const size_t count)
        : Bezier() {
        for (size_t i = 0; i < count; i++) {
            Curve<TVAL, TPARAM>::m_Knots.push_back(knots[i]);
            AddControlPoints(knots[i], incoming_cp[i], outgoing_cp[i]);
        }
    }

    void AddControlPoints(const TVAL knot, const TVAL incoming_cp,
                          const TVAL outgoing_cp) {
        m_IncomingControlPoints.insert({knot, incoming_cp});
        m_OutgoingControlPoints.insert({knot, outgoing_cp});
    }

    [[nodiscard]] TPARAM Reverse(TVAL t, size_t& index) const final {
        TVAL t1{0};
        TVAL t2{0};

        index = 0;

        if (Curve<TVAL, TPARAM>::m_Knots.size() < 2) {
            return TPARAM(0);
        }

        if (t <= Curve<TVAL, TPARAM>::m_Knots.front()) {
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

        TVAL c1, c2;
        c1 = m_OutgoingControlPoints.find(t1)->second;
        c2 = m_IncomingControlPoints.find(t2)->second;

        typename NewtonRapson<TVAL, TPARAM>::nr_f f = [t2, t1, c2, c1,
                                                       t](TPARAM s) {
            return (t2 - 3.0f * c2 + 3.0f * c1 - t1) * pow(s, 3.0f) +
                   3.0f * (c2 - 2.0f * c1 + t1) * pow(s, 2.0f) +
                   3.0f * (c1 - t1) * s + t1 - t;
        };

        typename NewtonRapson<TVAL, TPARAM>::nr_fprime fprime = [t2, t1, c2,
                                                                 c1](TPARAM s) {
            return 3.0f * (t2 - 3.0f * c2 + 3.0f * c1 - t1) * pow(s, 2.0f) +
                   6.0f * (c2 - 2.0f * c1 + t1) * s + 3.0f * (c1 - t1);
        };

        return NewtonRapson<TVAL, TPARAM>::Solve(TPARAM(0.5f), f, fprime);
    }

    [[nodiscard]] TVAL Interpolate(TPARAM s, const size_t index) const final {
        if (Curve<TVAL, TPARAM>::m_Knots.empty()) return TVAL(0);

        if (Curve<TVAL, TPARAM>::m_Knots.size() == 1)
            return Curve<TVAL, TPARAM>::m_Knots[0];
        else if (Curve<TVAL, TPARAM>::m_Knots.size() < index + 1)
            return Curve<TVAL, TPARAM>::m_Knots.back();
        else if (index == 0)
            return Curve<TVAL, TPARAM>::m_Knots.front();
        else {
            auto t1 = Curve<TVAL, TPARAM>::m_Knots[index - 1];
            auto t2 = Curve<TVAL, TPARAM>::m_Knots[index];

            TVAL c1, c2;
            c1 = m_OutgoingControlPoints.find(t1)->second;
            c2 = m_IncomingControlPoints.find(t2)->second;

            return (t2 - 3.0f * c2 + 3.0f * c1 - t1) * pow(s, 3.0f) +
                   3.0f * (c2 - 2.0f * c1 + t1) * pow(s, 2.0f) +
                   3.0f * (c1 - t1) * s + t1;
        }
    }
};

template <typename T>
class Bezier<Quaternion<T>, T> : public CurveBase,
                                 public Curve<Quaternion<T>, T> {
   private:
    std::map<Quaternion<T>, Quaternion<T>> m_IncomingControlPoints;
    std::map<Quaternion<T>, Quaternion<T>> m_OutgoingControlPoints;

   public:
    Bezier() : CurveBase(CurveType::kBezier) {}
    Bezier(const std::vector<Quaternion<T>> knots,
           const std::vector<Quaternion<T>> incoming_cp,
           const std::vector<Quaternion<T>> outgoing_cp)
        : Bezier() {
        assert(knots.size() == incoming_cp.size() &&
               knots.size() == outgoing_cp.size());

        Curve<Quaternion<T>, T>::m_Knots = knots;

        auto count = knots.size();

        for (size_t i = 0; i < count; i++) {
            auto knot = knots[i];
            auto in_cp = incoming_cp[i];
            auto out_cp = outgoing_cp[i];
            AddControlPoints(knot, in_cp, out_cp);
        }
    }

    Bezier(const Quaternion<T>* knots, const Quaternion<T>* incoming_cp,
           const Quaternion<T>* outgoing_cp, const size_t count)
        : Bezier() {
        for (size_t i = 0; i < count; i++) {
            Curve<Quaternion<T>, T>::m_Knots.push_back(knots[i]);
            AddControlPoints(knots[i], incoming_cp[i], outgoing_cp[i]);
        }
    }

    void AddControlPoints(const Quaternion<T>& knot,
                          const Quaternion<T>& incoming_cp,
                          const Quaternion<T>& outgoing_cp) {
        assert(incoming_cp <= knot && knot <= outgoing_cp);
        m_IncomingControlPoints.insert({knot, incoming_cp});
        m_OutgoingControlPoints.insert({knot, outgoing_cp});
    }

    T Reverse(Quaternion<T> t, size_t& index) const final {
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
class Bezier<Matrix4X4f, float> : public CurveBase,
                                  public Curve<Matrix4X4f, float> {
   private:
    std::map<Matrix4X4f, Matrix4X4f> m_IncomingControlPoints;
    std::map<Matrix4X4f, Matrix4X4f> m_OutgoingControlPoints;

   public:
    Bezier() : CurveBase(CurveType::kBezier) {}
    Bezier(const std::vector<Matrix4X4f>& knots,
           const std::vector<Matrix4X4f>& incoming_cp,
           const std::vector<Matrix4X4f>& outgoing_cp)
        : Bezier() {
        assert(knots.size() == incoming_cp.size() &&
               knots.size() == outgoing_cp.size());

        Curve<Matrix4X4f, float>::m_Knots = knots;

        auto count = knots.size();

        for (size_t i = 0; i < count; i++) {
            auto knot = knots[i];
            auto in_cp = incoming_cp[i];
            auto out_cp = outgoing_cp[i];
            AddControlPoints(knot, in_cp, out_cp);
        }
    }

    Bezier(const Matrix4X4f* knots, const Matrix4X4f* incoming_cp,
           const Matrix4X4f* outgoing_cp, const size_t count)
        : Bezier() {
        for (size_t i = 0; i < count; i++) {
            Curve<Matrix4X4f, float>::m_Knots.push_back(knots[i]);
            AddControlPoints(knots[i], incoming_cp[i], outgoing_cp[i]);
        }
    }

    void AddControlPoints(const Matrix4X4f& knot, const Matrix4X4f& incoming_cp,
                          const Matrix4X4f& outgoing_cp) {
        assert(incoming_cp <= knot && knot <= outgoing_cp);
        m_IncomingControlPoints.insert({knot, incoming_cp});
        m_OutgoingControlPoints.insert({knot, outgoing_cp});
    }

    float Reverse(Matrix4X4f t, size_t& index) const final {
        float result = 0.0f;
        assert(0);

        return result;
    }

    [[nodiscard]] Matrix4X4f Interpolate(float s,
                                         const size_t index) const final {
        Matrix4X4f result;
        BuildIdentityMatrix(result);
        assert(0);

        return result;
    }
};
}  // namespace My
