export void Normalize(uniform const size_t count, uniform float v[], uniform float length)
{
    length = 1.0f / length;
    foreach (index = 0 ... count) {
        v[index] *= length;
    }
}
