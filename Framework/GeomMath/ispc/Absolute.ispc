export void Absolute(uniform float result[], uniform const float a[], uniform const size_t count)
{
    foreach (index = 0 ... count)
    {
        result[index] = abs(a[index]);
    }
}