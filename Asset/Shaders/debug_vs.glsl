#version 150

/////////////////////
// INPUT VARIABLES //
/////////////////////
in vec3 inputPosition;

///////////////////////
// UNIFORM VARIABLES //
// update per frame
uniform mat4 worldMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void main(void)
{
	// Some configuration
	gl_PointSize = 5.0f;

	// Calculate the position of the vertex against the world, view, and projection matrices.
	vec4 v = worldMatrix * vec4(inputPosition, 1.0f);
	v = viewMatrix * v;
	gl_Position = projectionMatrix * v;
}
