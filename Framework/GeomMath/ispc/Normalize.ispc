extern void DotProduct(uniform const float a[], uniform const float b[], uniform float* uniform result, uniform const size_t count);

export void Normalize(uniform float v[], const uniform size_t count)
{
    uniform float length;
    DotProduct(v, v, &length, count);
    length = 1.0f / sqrt(length);
    foreach (index = 0 ... count) {
        v[index] *= length;
    }
}
