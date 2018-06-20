#include <iostream>
#include <limits>
#include <cassert>
#include <string>

using namespace std;

bool is_big_endian()
{
    uint16_t test_num = 0xFF00;

    return reinterpret_cast<uint8_t*>(&test_num)[0] == 0xFF;
}

template <typename T>
void dump_binary(T* pNumber)
{
    uint8_t* p = reinterpret_cast<uint8_t*>(pNumber);
    uint8_t format[2];

    switch(sizeof(T))
    {
        case 4: // float
            format[0] = 1; // 1 bit sign
            format[1] = 9; // 8 bit exponent
            break;
        case 8: // double
            format[0] = 1; // 1 bit sign
            format[1] = 12; // 11 bit exponent
            break;
        default:
            assert(0);
    }

    int32_t n = 0;

    if(is_big_endian())
    {
        for (int32_t i = 0; i < sizeof(T); i++)
        {
           int32_t j = 0;
           while(j++ < 8)
           {
               cout << ((p[i] & 0x80)?'1':'0');
               p[i] <<= 1;
               // format beautifier 
               n++;
               if(n == format[0] || n == format[1])
                   cout << ' ';
           }
        }
    }
    else
    {
        for (int32_t i = sizeof(T) - 1; i >= 0; i--)
        {
           int32_t j = 0;
           while(j++ < 8)
           {
               cout << ((p[i] & 0x80)?'1':'0');
               p[i] <<= 1;
               // format beautifier 
               n++;
               if(n == format[0] || n == format[1])
                   cout << ' ';
           }
        }
    }
    cout << endl;
}

int main(int argc, char** argv)
{
    if (argc == 1)
    {
        cout << "Usage: dump_float_as_binary [float|double] num1 [[float|double] num2]] [...]" << endl;
        return 0;
    }

    bool is_double = true;

    for (int i = 1; i < argc; i++)
    {
        string str (argv[i]);

        if (str == "float")
        {
            is_double = false;
            continue;
        }
        else if (str == "double")
        {
            is_double = true;
            continue;
        }

        double tmp = atof(argv[i]);
        if (!is_double)
        {
            float _tmp = static_cast<float>(tmp);
            dump_binary(&_tmp);
        }
        else
        {
            dump_binary(&tmp);
        }
    }

    return 0;
}
