export void MulByElement(uniform const float a[], uniform const float b[], uniform float result[], uniform const size_t count)
{
	foreach (index = 0 ... count) {
		result[index] = a[index] * b[index];
	}
}

export void MulByElementi32(uniform const int32 a[], uniform const int32 b[], uniform int32 result[], uniform const size_t count)
{
	foreach (index = 0 ... count) {
		result[index] = a[index] * b[index];
	}
}

export void MulByElementi16(uniform const int16 a[], uniform const int16 b[], uniform int16 result[], uniform const size_t count)
{
	foreach (index = 0 ... count) {
		result[index] = a[index] * b[index];
	}
}

