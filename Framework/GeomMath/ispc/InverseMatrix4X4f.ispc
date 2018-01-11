float invf(int i, int j, uniform const float matrix[16])
{
	int pre_i = ((i == 0) ? 3 : i - 1);
	int next_i = ((i + 1 == 4) ? 0 : i + 1);
	int next_next_i = ((i + 2 >= 4) ? i - 2 : i + 2);
	int pre_j = ((j == 0) ? 3 : j - 1);
	int next_j = ((j + 1 == 4) ? 0 : j + 1);
	int next_next_j = ((j + 2 >= 4) ? j - 2 : j + 2);
    int o = i-j < 0? j-i:i-j;

#define e(a, b) matrix[(a)*4 + (b)]
    float inv =
        + e(next_i, next_j)*e(next_next_i, next_next_j)*e(pre_i, pre_j)
        + e(next_i, next_next_j)*e(next_next_i, pre_j)*e(pre_i, next_j)
        + e(next_i, pre_j)*e(next_next_i, next_j)*e(pre_i, next_next_j)
        - e(next_i, next_j)*e(next_next_i, pre_j)*e(pre_i, next_next_j)
        - e(next_i, next_next_j)*e(next_next_i, next_j)*e(pre_i, pre_j)
        - e(next_i, pre_j)*e(next_next_i, next_next_j)*e(pre_i, next_j);

    return (o&0x1)?-inv : inv;
#undef e
}

export uniform bool InverseMatrix4X4f(uniform float matrix[16])
{
    uniform float inv[16];
    uniform double D = 0;

    foreach_tiled(i = 0 ... 4, j = 0 ... 4) {
        inv[j*4+i] = invf(i,j,matrix);
    }

    foreach(k = 0 ... 4) {
        float s = matrix[k] * inv[k*4];
        D = reduce_add(s);
    }

    if (D==0) return false;

    D = 1.0 / D;

    foreach(i = 0 ... 16) {
        matrix[i] = inv[i] * D;
    }

    return true;
}
