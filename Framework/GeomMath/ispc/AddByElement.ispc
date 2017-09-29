export void AddByElement(uniform const float a[], uniform const float b[], uniform float result[], uniform const size_t count)
{
    foreach(index = 0 ... count)
    {
        result[index] = a[index] + b[index];
    }

    return;
}

