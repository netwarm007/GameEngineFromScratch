export void TransformCoord(uniform float vector[3], uniform const float matrix[9])
{
    foreach(index = 0 ... 3) {
        vector[index] = (vector[0] * matrix[index]) 
            + (vector[1] * matrix[index + 3]) 
            + (vector[2] * matrix[index + 6]);
    }

    return;
}

