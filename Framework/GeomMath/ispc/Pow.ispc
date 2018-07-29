export void Pow(uniform const float v[], uniform const size_t count, uniform const float exponent, uniform float result[])
{
    foreach (index = 0 ... count) {
        result[index] = pow(v[index], exponent);
    }
}