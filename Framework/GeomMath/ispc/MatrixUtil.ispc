export void MatrixExchangeYandZ(uniform float data[], const uniform int rows, const uniform int cols)
{
    foreach(row_index = 0 ... rows) {
        int start_index = row_index * cols;
        unsigned int* p = (unsigned int*)(data + start_index + 1);
        unsigned int* q = (unsigned int*)(data + start_index + 2);
        *p ^= *q;
        *q ^= *p;
        *p ^= *q;
    }
}

export void BuildIdentityMatrix(uniform float data[], const uniform int n)
{
	foreach(i = 0 ... n, j = 0 ... n) {
		if (i == j)
			data[i * n + j] = 1;
		else
			data[i * n + j] = 0;
	}
}

