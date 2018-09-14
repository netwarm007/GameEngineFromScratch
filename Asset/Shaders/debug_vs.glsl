/////////////////////
// INPUT VARIABLES //
/////////////////////
layout(location = 0) in vec3 inputPosition;

////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void main(void)
{
	// Some configuration
	gl_PointSize = 5.0f;

	// Calculate the position of the vertex against the world, view, and projection matrices.
	vec4 v = modelMatrix * vec4(inputPosition, 1.0f);
	v = viewMatrix * v;
	gl_Position = projectionMatrix * v;
}
