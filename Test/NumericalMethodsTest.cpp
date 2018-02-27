#include <iostream>
#include <iomanip>
#include "numerical.hpp"

using namespace My;
using namespace std;

int main(int, char**)
{
    cout.precision(4);
    cout.setf(ios::fixed);
    nr_f f = [](double x) { return pow(x, 3.0) - x - 11.0; };
    nr_fprime fprime = [](double x) { return 3 * pow(x, 2.0) - 1.0; };
    auto result = newton_raphson(2, f, fprime);
    cout << "root of equation x^3 - x - 11 = 0 is: " << result << endl;

    return 0;
}