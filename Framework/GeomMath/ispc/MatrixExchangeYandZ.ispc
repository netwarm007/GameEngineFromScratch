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

