export void Transform(uniform float vector[4], uniform const float matrix[16])
{
    uniform const float x = vector[0];
    uniform const float y = vector[1];
    uniform const float z = vector[2];
    uniform const float w = vector[3];
    foreach(index = 0 ... 4) {
        vector[index] = (x * matrix[index])
            + (y * matrix[index + 4])
            + (z * matrix[index + 8])
            + (w * matrix[index + 12]);
    }
}

