export void Transpose(uniform float a[], uniform float r[], uniform int row_count, uniform int column_count) {
	foreach(i = 0 ... row_count, j = 0 ... column_count) {
		r[j*row_count+i] = a[i*column_count+j];
	}
}

