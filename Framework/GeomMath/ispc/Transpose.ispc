export void Transpose(uniform const float a[], uniform float r[], uniform const size_t row_count, uniform const size_t column_count) {
	foreach(i = 0 ... row_count, j = 0 ... column_count) {
		r[j*row_count+i] = a[i*column_count+j];
	}
}

