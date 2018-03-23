#include <iostream>
#include <iomanip>
#include "numerical.hpp"

using namespace My;
using namespace std;

int main(int argc, char** argv)
{
    double t1 = 0, t2 = 15;
    double c1 = 4, c2 = 9;
    double t;
    cout.precision(4);
    cout.setf(ios::fixed);
    for (int p = 0; p <=100; p++)
    {
        t = p / 100.0 * (t2 - t1) + t1;
        nr_f f = [t2, t1, c2, c1, t](double s) { 
            return (t2 - 3 * c2 + 3 * c1 - t1) * pow(s, 3.0) 
                + 3 * (c2 - 2 * c1 + t1) * pow(s, 2.0)
                + 3 * (c1 - t1) * s 
                + t1 - t; 
        };
        nr_fprime fprime = [t2, t1, c2, c1](double s) {
            return 3 * (t2 - 3 * c2 + 3 * c1 - t1) * pow(s, 2.0) 
                + 6 * (c2 - 2 * c1 + t1) * s
                + 3 * (c1 - t1);
        };
        auto result = newton_raphson(0.5, f, fprime);
        cout << t << ": " << result << endl;
    }

    return 0;
}
