#pragma once
#include <cmath>
#include <functional>
#include <limits>

using namespace std;

namespace My {
template <typename TVAL, typename TPARAM>
struct NewtonRapson {
    using nr_f = std::function<TVAL(TPARAM)>;
    using nr_fprime = std::function<TVAL(TPARAM)>;

    static inline TPARAM Solve(TPARAM x0, nr_f f, nr_fprime fprime) {
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
}  // namespace My
