export void CrossProduct(uniform float a[3], uniform float b[3], uniform float result[3]) {
	foreach(index = 0 ... 3) {
		int index_a = ((index + 1 == 3) ? 0 : index + 1);
		int index_b = ((index == 0) ? 2 : index - 1);
		result[index] = a[index_a] * b[index_b]
				-a[index_b] * b[index_a];
	}
}
