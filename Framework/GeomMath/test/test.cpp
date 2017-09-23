#include <iostream>
#include "geommath.hpp"

using namespace std;
using namespace My;

int main()
{
    Vector3f a = { 1.0f, 2.0f, 3.0f };
	Vector3f b = { 5.0f, 6.0f, 7.0f };
	Vector3f c;

	cout << a;
	cout << b;
    CrossProduct(c, a, b);
	cout << c;

	return 0;
}

