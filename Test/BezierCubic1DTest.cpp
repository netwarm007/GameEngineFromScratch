#include <iostream>
#include <iomanip>
#include "Bezier.hpp"

using namespace My;
using namespace std;

int main(int argc, char** argv)
{
    float t;
    Bezier<float> bezier1({0.0f, 2.4583f}, {-0.9598f, 0.9598f}, {1.4986f, 3.4181f});
    Bezier<float> bezier2({0.0f, 1.3821f}, {0.0f, 0.0f}, {1.3821f, 1.3821f});

    cout.precision(4);
    cout.setf(ios::fixed);
    for (int p = 0; p <=100; p++)
    {
        t = p / 100.0f * (2.4583f - 0.0f) + 0.0f;
        auto result = bezier2.Interpolate(bezier1.Reverse(t));
        cout << t << ": " << result << endl;
    }

    return 0;
}
