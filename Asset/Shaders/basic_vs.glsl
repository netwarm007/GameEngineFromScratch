////////////////////////////////////////////////////////////////////////////////
// Filename: basic.vs
////////////////////////////////////////////////////////////////////////////////

/////////////////////
// INPUT VARIABLES //
/////////////////////
layout(location = 0) in vec3 inputPosition;
layout(location = 1) in vec3 inputNormal;
layout(location = 2) in vec2 inputUV;

//////////////////////
// OUTPUT VARIABLES //
//////////////////////
layout(location = 0) out vec4 normal;
layout(location = 1) out vec4 normal_world;
layout(location = 2) out vec4 v;
layout(location = 3) out vec4 v_world;
layout(location = 4) out vec2 uv;

////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void main(void)
{
	// Calculate the position of the vertex against the world, view, and projection matrices.
	v_world = modelMatrix * vec4(inputPosition, 1.0f);
	v = viewMatrix * v_world;
	gl_Position = projectionMatrix * v;

    normal_world = modelMatrix * vec4(inputNormal, 0.0f);
    normal = viewMatrix * normal_world;
    uv.x = inputUV.x;
    uv.y = 1.0f - inputUV.y;
}

