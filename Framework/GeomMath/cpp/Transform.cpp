namespace Dummy
{
    void Transform(float vector[4], const float matrix[16])
    {
        for(int index = 0; index < 4; index++)
        {
            vector[index] = (vector[0] * matrix[index]) 
                + (vector[1] * matrix[index + 4]) 
                + (vector[2] * matrix[index + 8]) 
                + (vector[3] * matrix[index + 12]);
        }
    }
}
