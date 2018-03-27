#include <iostream>
#include <iomanip>
#include "Bezier.hpp"

using namespace My;
using namespace std;

int main(int argc, char** argv)
{
    float t;
    Bezier<float> bezier1({0, 2.4583}, {-0.9598, 0.9598}, {1.4986, 3.4181});
    Bezier<float> bezier2({0, 1.3821}, {0, 0}, {1.3821, 1.3821});

    cout.precision(4);
    cout.setf(ios::fixed);
    for (int p = 0; p <=100; p++)
    {
        t = p / 100.0 * (2.4583 - 0) + 0;
        auto result = bezier2.Interpolate(bezier1.Reverse(t));
        cout << t << ": " << result << endl;
    }

    return 0;
}
