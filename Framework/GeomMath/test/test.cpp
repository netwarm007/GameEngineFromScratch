#include "CrossProduct.h"
#include <iostream>

using namespace std;

int main()
{
	float a[] = { 1, 2, 3 };
	float b[] = { 5, 6, 7 };
	float c[3];
	ispc::CrossProduct(a, b, c);
	cout << c[0] << "," << c[1] << "," << c[2] << endl;

	return 0;
}

