layout(location = 0) in vec3 inputPosition;

out vec3 UVW;

void main()
{
    UVW = inputPosition.xyz;
    mat4 matrix = viewMatrix;
    matrix[3][0] = 0.0f;
    matrix[3][1] = 0.0f;
    matrix[3][2] = 0.0f;
	vec4 pos = projectionMatrix * matrix * vec4(inputPosition, 1.0f);
    gl_Position = pos.xyww;
}  
