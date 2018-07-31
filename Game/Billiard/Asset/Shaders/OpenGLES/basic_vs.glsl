/////////////////////
// INPUT VARIABLES //
/////////////////////
in vec3 inputPosition;
in vec3 inputNormal;
in vec2 inputUV;

//////////////////////
// OUTPUT VARIABLES //
//////////////////////
out vec4 normal;
out vec4 v;
out vec2 uv;

////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void main(void)
{
    mat4 transformMatrix = worldMatrix * modelMatrix;
	// Calculate the position of the vertex against the world, view, and projection matrices.
	v = transformMatrix * vec4(inputPosition, 1.0f);
	v = viewMatrix * v;
	gl_Position = projectionMatrix * v;

    normal = transformMatrix * vec4(inputNormal, 0.0f);
    normal = viewMatrix * normal;
    uv.x = inputUV.x;
    uv.y = 1.0f - inputUV.y;
}
