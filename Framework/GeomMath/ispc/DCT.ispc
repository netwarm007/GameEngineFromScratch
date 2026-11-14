uniform const float dct_basis[8][8] = {
#include "../DCTBasis.inc"
};

export void DCT8X8(uniform const float g[64], uniform float G[64])
{
    uniform float intermediate[64];
    for (uniform int u = 0; u < 8; u++) {
        foreach(y = 0 ... 8) {
            float sum = 0.0f;
            for (uniform int x = 0; x < 8; x++) {
                sum += dct_basis[u][x] * g[x * 8 + y];
            }
            intermediate[y * 8 + u] = sum;
        }
    }
    for (uniform int v = 0; v < 8; v++) {
        foreach(u = 0 ... 8) {
            float sum = 0.0f;
            for (uniform int y = 0; y < 8; y++) {
                sum += intermediate[y * 8 + u] * dct_basis[v][y];
            }
            G[u * 8 + v] = sum;
        }
    }
}

export void IDCT8X8(uniform const float G[64], uniform float g[64])
{
    uniform float intermediate[64];
    foreach(x = 0 ... 8, v = 0 ... 8) {
        float sum = 0.0f;
        for (uniform int u = 0; u < 8; u++) {
            sum += dct_basis[u][x] * G[u * 8 + v];
        }
        intermediate[x * 8 + v] = sum;
    }
    foreach(x = 0 ... 8, y = 0 ... 8) {
        float sum = 0.0f;
        for (uniform int v = 0; v < 8; v++) {
            sum += intermediate[x * 8 + v] * dct_basis[v][y];
        }
        g[x * 8 + y] = sum;
    }
}
