float invf3X3(int i, int j, uniform const float matrix[9])
{
	int pre_i = ((i == 0) ? 2 : i - 1);
	int next_i = ((i + 1 == 3) ? 0 : i + 1);
	int pre_j = ((j == 0) ? 2 : j - 1);
	int next_j = ((j + 1 == 3) ? 0 : j + 1);

#define e(a, b) matrix[(a)*3 + (b)]
    float inv =
        + e(next_i, next_j)*e(pre_i, pre_j)
        - e(next_i, pre_j)*e(pre_i, next_j);

    return inv;
#undef e
}

float invf4X4(int i, int j, uniform const float matrix[16])
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

export uniform bool InverseMatrix3X3f(uniform float matrix[9])
{
    uniform float inv[9];
    uniform double D = 0;

    foreach_tiled(i = 0 ... 3, j = 0 ... 3) {
        inv[j*3+i] = invf3X3(i,j,matrix);
    }

    foreach(k = 0 ... 3) {
        float s = matrix[k] * inv[k*3];
        D = reduce_add(s);
    }

    if (D==0) return false;

    D = 1.0 / D;

    foreach(i = 0 ... 9) {
        matrix[i] = inv[i] * D;
    }

    return true;
}

export uniform bool InverseMatrix4X4f(uniform float matrix[16])
{
    uniform float inv[16];
    uniform double D = 0;

    foreach_tiled(i = 0 ... 4, j = 0 ... 4) {
        inv[j*4+i] = invf4X4(i,j,matrix);
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
