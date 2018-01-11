uniform const float one_over_four = 1.0f / 4.0f;
uniform const float PI_over_sixteen = PI / 16.0f;

inline float normalizing_scale_factor(float a)
{
    return (a == 0)? 1.0f/sqrt(2.0f) : 1.0f;
}

export void DCT8X8(uniform const float g[64], uniform float G[64])
{
    uniform float result_cache[8][8][8][8];

    foreach(u = 0 ... 8, v = 0 ... 8, x = 0 ... 8, y = 0 ... 8) {
        float scale_factor = one_over_four * normalizing_scale_factor(u)
                * normalizing_scale_factor(v);
        float normial = g[x*8+y] * cos((2.0f * x + 1.0f) * u * PI_over_sixteen)
                * cos((2.0f * y + 1.0f) * v * PI_over_sixteen);
        result_cache[u][v][x][y] = scale_factor * normial;
    }

    foreach(u = 0 ... 8, v = 0 ... 8) {
        G[u*8+v] = 0;
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                G[u*8+v] += result_cache[u][v][x][y];
            }
        }
    }
}

export void IDCT8X8(uniform const float G[64], uniform float g[64])
{
    uniform float result_cache[8][8][8][8];

    foreach(x = 0 ... 8, y = 0 ... 8, u = 0 ... 8, v = 0 ... 8) {
        float scale_factor = one_over_four * normalizing_scale_factor(u)
            * normalizing_scale_factor(v);
        float normial = G[u*8+v] * cos((2.0f * x + 1.0f) * u * PI_over_sixteen)
            * cos((2.0f * y + 1.0f) * v * PI_over_sixteen);
        result_cache[x][y][u][v] = scale_factor * normial;
    }

    foreach(x = 0 ... 8, y = 0 ... 8) {
        float sum = 0.0f;
        for (int u = 0; u < 8; u++) {
            for (int v = 0; v < 8; v++) {
                sum += result_cache[x][y][u][v];
            }
        }
        g[x*8+y] = sum;
    }
}

