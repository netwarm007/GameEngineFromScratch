export void DotProduct(uniform const float a[], uniform const float b[], uniform float result[], uniform const size_t count)
{
	foreach (index = 0 ... count) {
        float _a = a[index];
        float _b = b[index];
        *result = reduce_add(_a * _b);
	}

}
