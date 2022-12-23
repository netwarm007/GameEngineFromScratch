export void Sqrt(uniform const float v[], uniform const size_t count, uniform float result[])
{
    foreach (index = 0 ... count) {
        result[index] = sqrt(v[index]);
    }
}