#include <iostream>
#include "ColorSpaceConversion.hpp"

using namespace std;
using namespace My;

int main(int argc, const char** argv)
{
    int result = 0;

    RGBf rgb = { 64, 35, 17 };
    cout << "RGB Color: " << rgb;
    YCbCrf ycbcr = ConvertRGB2YCbCr(rgb);
    cout << "When transformed to YCbCr: " << ycbcr;
    rgb = ConvertYCbCr2RGB(ycbcr);
    cout << "Now transformed back to RGB: " << rgb;

    return result;
}

