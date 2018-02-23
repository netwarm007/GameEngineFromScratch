#pragma once
#include <functional>
#include <limits>
#include <cmath>

namespace My {
    typedef std::function<double(double)> nr_f;
    typedef std::function<double(double)> nr_fprime;

    inline double newton_raphson(double x0, nr_f f, nr_fprime fprime)
    {
        double x, x1 = x0;

        do {
            x = x1;
            double fx = f(x);
            double fx1 = fprime(x);
            x1 = x-(fx/fx1);
        } while (fabs(x1 - x) >= std::numeric_limits<double>::epsilon());

        return x1;
    }
}