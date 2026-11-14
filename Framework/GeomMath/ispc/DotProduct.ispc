export void DotProduct(uniform const float a[], uniform const float b[], uniform float result[], uniform const size_t count)
{
    float sum = 0.0f;
    foreach (index = 0 ... count) {
        sum += a[index] * b[index];
    }
    *result = reduce_add(sum);
}
