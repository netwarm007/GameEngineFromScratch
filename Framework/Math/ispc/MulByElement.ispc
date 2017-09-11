export void MulByElement(uniform float a[], uniform float b[], uniform float result[], uniform int count)
{
	foreach (index = 0 ... count) {
		result[index] = a[index] * b[index];
	}
}

