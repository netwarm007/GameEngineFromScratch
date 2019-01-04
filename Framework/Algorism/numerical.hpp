#pragma once
#include <functional>
#include <limits>
#include <cmath>

using namespace std;

namespace My {
    template <typename TVAL, typename TPARAM>
    struct NewtonRapson
    {
        typedef std::function<TVAL(TPARAM)> nr_f;
        typedef std::function<TVAL(TPARAM)> nr_fprime;

        static inline TPARAM Solve(TPARAM x0, nr_f f, nr_fprime fprime)
        {
            TPARAM x, x1 = x0;

            do {
                x = x1;
                TVAL fx = f(x);
                TVAL fx1 = fprime(x);
                x1 = x - (fx / fx1);
            } while (fabs(x1 - x) >= 10E-6f);

            return x1;
        }
    };
}
