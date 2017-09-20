#include "MulByElement.isph"

export void DotProduct(uniform const float a[], uniform const float b[], uniform float* uniform result, uniform const int count) {
	*result = 0;
	uniform float * uniform r = uniform new uniform float [count];
	MulByElement(a, b, r, count);
	foreach_active(i) {
		*result += r[i];
	}
	delete[] r;
}

