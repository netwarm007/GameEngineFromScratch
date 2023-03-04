#include <iomanip>
#include <iostream>

#include "numerical.hpp"

using namespace My;
using namespace std;

int main(int, char**) {
    cout.precision(4);
    cout.setf(ios::fixed);
    NewtonRapson<double, double>::nr_f f = [](double x) {
        return pow(x, 3.0) - x - 11.0;
    };
    NewtonRapson<double, double>::nr_fprime fprime = [](double x) {
        return 3 * pow(x, 2.0) - 1.0;
    };
    auto result = NewtonRapson<double, double>::Solve(2, f, fprime);
    cout << "root of equation x^3 - x - 11 = 0 is: " << result << endl;

    return 0;
}