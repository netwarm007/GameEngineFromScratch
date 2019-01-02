#include <math.h>

#define PI 3.14159265354f
const float one_over_four = 1.0f / 4.0f;
const float PI_over_sixteen = PI / 16.0f;

inline float normalizing_scale_factor(float a)
{
    return (a == 0)? 1.0f/sqrt(2.0f) : 1.0f;
}

namespace Dummy
{
    void DCT8X8(const float g[64], float G[64])
    {
        for(int u = 0; u < 8; u++)
        {
            for(int v = 0; v < 8; v++)
            {
                G[u*8+v] = 0;

                for (int x = 0; x < 8; x++)
                {
                    for (int y = 0; y < 8; y++)
                    {
                        float scale_factor = one_over_four * normalizing_scale_factor(u)
                                * normalizing_scale_factor(v);
                        float normial = g[x*8+y] * cos((2.0f * x + 1.0f) * u * PI_over_sixteen)
                                * cos((2.0f * y + 1.0f) * v * PI_over_sixteen);

                        G[u*8+v] += scale_factor * normial;
                    }
                }
            }
        }
    }

    void IDCT8X8(const float G[64], float g[64])
    {
        for(int x = 0; x < 8; x++)
        {
            for(int y = 0; y < 8; y++)
            {
                g[x*8+y] = 0;

                for (int u = 0; u < 8; u++)
                {
                    for (int v = 0; v < 8; v++)
                    {
                        float scale_factor = one_over_four * normalizing_scale_factor(u)
                            * normalizing_scale_factor(v);
                        float normial = G[u*8+v] * cos((2.0f * x + 1.0f) * u * PI_over_sixteen)
                            * cos((2.0f * y + 1.0f) * v * PI_over_sixteen);

                        g[x*8+y] += scale_factor * normial;
                    }
                }
            }
        }
    }
}
