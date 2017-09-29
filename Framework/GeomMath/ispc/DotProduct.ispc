extern void MulByElement(uniform const float a[], uniform const float b[], uniform float result[], uniform const size_t count);

export void DotProduct(uniform const float a[], uniform const float b[], uniform float* uniform result, uniform const size_t count) {
	*result = 0;
	uniform float * uniform r = uniform new uniform float [count];
	MulByElement(a, b, r, count);
	foreach_active(i) {
		*result += r[i];
	}
	delete[] r;
}

