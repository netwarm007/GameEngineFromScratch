export void Normalize(uniform float v[], uniform float length, const uniform size_t count)
{
    length = 1.0f / length;
    foreach (index = 0 ... count) {
        v[index] *= length;
    }
}
