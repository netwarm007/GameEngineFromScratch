#pragma once
#include "Curve.hpp"
#include "numerical.hpp"
#include <cassert>
#include <map>

namespace My {
    template <typename T>
    class Bezier : public Curve<T>
    {
    private:
        std::map<T, T> m_IncomingControlPoints;
        std::map<T, T> m_OutgoingControlPoints;
        std::vector<T> m_Knots;

    public:
        Bezier() = default;
        Bezier(const std::vector<T> knots, const std::vector<T> incoming_cp, const std::vector<T> outgoing_cp) 
        {
            assert(knots.size() == incoming_cp.size() && knots.size() == outgoing_cp.size());

            m_Knots = knots;

            auto count = knots.size();

            for (auto i = 0; i < count; i++)
            {
                auto knot = knots[i];
                auto in_cp = incoming_cp[i];
                auto out_cp = outgoing_cp[i];
                AddControlPoints(knot, in_cp, out_cp);
            }
        }

        Bezier(const T* knots, const T* incoming_cp, const T* outgoing_cp, const size_t count)
        {
            for (auto i = 0; i < count; i++)
            {
                m_Knots.push_back(knots[i]);
                AddControlPoints(knots[i], incoming_cp[i], outgoing_cp[i]);
            }
        }

        void AddKnot(const T knot) 
        {
            m_Knots.push_back(knot);
        }

        void AddControlPoints(const T knot, const T incoming_cp, const T outgoing_cp) 
        {
            assert(incoming_cp <= knot && knot <= outgoing_cp);
            m_IncomingControlPoints.insert({knot, incoming_cp});
            m_OutgoingControlPoints.insert({knot, outgoing_cp});
        }

        T Reverse(T t) const final
        {
            T t1, t2;

            if (m_Knots.size() < 2)
                return 0;

            if (t <= m_Knots.front())
                return 0;

            if (t >= m_Knots.back())
                return 1;

            for (size_t i = 1; i < m_Knots.size(); i++)
            {
                if (t >= m_Knots[i])
                    continue;

                t1 = m_Knots[i - 1];                
                t2 = m_Knots[i];
            }

            T c1, c2;
            c1 = m_OutgoingControlPoints.find(t1)->second;
            c2 = m_IncomingControlPoints.find(t2)->second;

            typename NewtonRapson<T>::nr_f f = [t2, t1, c2, c1, t](T s) { 
                return (t2 - 3 * c2 + 3 * c1 - t1) * pow(s, 3.0) 
                    + 3 * (c2 - 2 * c1 + t1) * pow(s, 2.0)
                    + 3 * (c1 - t1) * s 
                    + t1 - t; 
            };

            typename NewtonRapson<T>::nr_fprime fprime = [t2, t1, c2, c1](T s) {
                return 3 * (t2 - 3 * c2 + 3 * c1 - t1) * pow(s, 2.0) 
                    + 6 * (c2 - 2 * c1 + t1) * s
                    + 3 * (c1 - t1);
            };

            return NewtonRapson<T>::Solve(0.5 * (t1 + t2), f, fprime);
        }

        T Interpolate(T s) const final
        {
            if (m_Knots.size() == 0)
                return 0;
            else if (m_Knots.size() == 1)
                return m_Knots[0];
            else
            {
                T t = s * (m_Knots.back() - m_Knots.front());
                T t1, t2;

                if (t <= m_Knots.front())
                    return m_Knots.front();

                if (t >= m_Knots.back())
                    return m_Knots.back();

                for (size_t i = 1; i < m_Knots.size(); i++)
                {
                    if (t >= m_Knots[i])
                        continue;

                    t1 = m_Knots[i - 1];                
                    t2 = m_Knots[i];
                }

                T c1, c2;
                c1 = m_OutgoingControlPoints.find(t1)->second;
                c2 = m_IncomingControlPoints.find(t2)->second;

                return (t2 - 3 * c2 + 3 * c1 - t1) * pow(s, 3.0) 
                    + 3 * (c2 - 2 * c1 + t1) * pow(s, 2.0)
                    + 3 * (c1 - t1) * s 
                    + t1; 
            }
        }

        CurveType GetCurveType() const final
        {
            return CurveType::kBezier;
        }
    };
}
