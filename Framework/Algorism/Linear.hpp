#pragma once
#include "Curve.hpp"

namespace My {
    template<typename TVAL, typename TPARAM>
    class Linear : public CurveBase, public Curve<TVAL, TPARAM> 
    {
    public:
        Linear() : CurveBase(CurveType::kLinear) {}
        Linear(const std::vector<TVAL> knots) 
            : Linear()
        {
            Curve<TVAL, TPARAM>::m_Knots = knots;
        }

        Linear(const TVAL* knots, const size_t count)
            : Linear()
        {
            for (auto i = 0; i < count; i++)
            {
                Curve<TVAL, TPARAM>::m_Knots.push_back(knots[i]);
            }
        }

        TPARAM Reverse(TVAL t, size_t& index) const final
        {
            TVAL t1 = 0, t2 = 0;

            if (Curve<TVAL, TPARAM>::m_Knots.size() < 2)
                return 0;

            if (t <= Curve<TVAL, TPARAM>::m_Knots.front())
            {
                index = 0;
                return 0;
            }

            if (t >= Curve<TVAL, TPARAM>::m_Knots.back())
            {
                index = Curve<TVAL, TPARAM>::m_Knots.size();
                return 1;
            }

            for (size_t i = 1; i < Curve<TVAL, TPARAM>::m_Knots.size(); i++)
            {
                if (t >= Curve<TVAL, TPARAM>::m_Knots[i])
                    continue;

                t1 = Curve<TVAL, TPARAM>::m_Knots[i - 1];                
                t2 = Curve<TVAL, TPARAM>::m_Knots[i];
                index = i;
                break;
            }

            return TPARAM((t - t1) / (t2 - t1));
        }

        TVAL Interpolate(TPARAM s, const size_t index) const final
        {
            if (Curve<TVAL, TPARAM>::m_Knots.size() == 0)
                return 0;
            else if (Curve<TVAL, TPARAM>::m_Knots.size() == 1)
                return Curve<TVAL, TPARAM>::m_Knots[0];
            else if (Curve<TVAL, TPARAM>::m_Knots.size() < index + 1)
                return Curve<TVAL, TPARAM>::m_Knots.back();
            else if (index == 0)
                return Curve<TVAL, TPARAM>::m_Knots.front();
            else
            {
                auto t1 = Curve<TVAL, TPARAM>::m_Knots[index - 1];                
                auto t2 = Curve<TVAL, TPARAM>::m_Knots[index];

                return t1 + s * (t2 - t1);
            }
        }
    };

    template<typename T>
    class Linear<Quaternion<T>, T> : public CurveBase, public Curve<Quaternion<T>, T> 
    {
    public:
        Linear() : CurveBase(CurveType::kLinear) {}
        Linear(const std::vector<Quaternion<T>> knots) 
            : Linear()
        {
            Curve<Quaternion<T>, T>::m_Knots = knots;
        }

        Linear(const Quaternion<T>* knots, const size_t count)
            : Linear()
        {
            for (auto i = 0; i < count; i++)
            {
                Curve<Quaternion<T>, T>::m_Knots.push_back(knots[i]);
            }
        }

        T Reverse(Quaternion<T> v, size_t& index) const final
        {
            T result = 0;
            assert(0);

            return result;
        }

        Quaternion<T> Interpolate(T s, const size_t index) const final
        {
            Quaternion<T> result;
            assert(0);

            return result;
        }
    };

    template<>
    class Linear<Matrix4X4f, float> : public CurveBase, public Curve<Matrix4X4f, float> 
    {
    public:
        Linear() : CurveBase(CurveType::kLinear) {}
        Linear(const std::vector<Matrix4X4f> knots) 
            : Linear()
        {
            Curve<Matrix4X4f, float>::m_Knots = knots;
        }

        Linear(const Matrix4X4f* knots, const size_t count)
            : Linear()
        {
            for (auto i = 0; i < count; i++)
            {
                Curve<Matrix4X4f, float>::m_Knots.push_back(knots[i]);
            }
        }

        float Reverse(Matrix4X4f v, size_t& index) const final
        {
            float result = 0.0f;
            assert(0);

            return result;
        }

        Matrix4X4f Interpolate(float s, const size_t index) const final
        {
            Matrix4X4f result;
            BuildIdentityMatrix(result);
            assert(0);

            return result;
        }
    };
}