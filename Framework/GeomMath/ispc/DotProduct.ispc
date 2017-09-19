#include "MulByElement.isph"

export void DotProduct(uniform float a[], uniform float b[], uniform float* uniform result, uniform int count) {
	*result = 0;
	uniform float * uniform r = uniform new uniform float [count];
	MulByElement(a, b, r, count);
	foreach_active(i) {
		*result += r[i];
	}
	delete[] r;
}

