#include <iostream>
#include "ColorSpaceConversion.hpp"

using namespace std;
using namespace My;

int main(int argc, const char** argv)
{
    int result = 0;

    RGB rgb = { 64, 35, 17 };
    cout << "RGB Color: " << rgb;
    YCbCr ycbcr = ConvertRGB2YCbCr(rgb);
    cout << "When transformed to YCbCr: " << ycbcr;
    rgb = ConvertYCbCr2RGB(ycbcr);
    cout << "Now transformed back to RGB: " << rgb;

    return result;
}

