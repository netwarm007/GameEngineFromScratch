export void Transform(uniform float vector[4], uniform const float matrix[16])
{
    foreach(index = 0 ... 4) {
        vector[index] = (vector[0] * matrix[index]) 
            + (vector[1] * matrix[index + 4]) 
            + (vector[2] * matrix[index + 8]) 
            + (vector[3] * matrix[index + 12]);
    }

    return;
}

