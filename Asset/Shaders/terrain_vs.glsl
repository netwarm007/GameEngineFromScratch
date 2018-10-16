layout(binding = 11) uniform sampler2D terrainHeightMap;

/////////////////////
// INPUT VARIABLES //
/////////////////////
layout(location = 0) in vec3 inputPosition;
layout(location = 1) in vec2 inputUV;

//////////////////////
// OUTPUT VARIABLES //
//////////////////////
layout(location = 0) out vec4 normal;
layout(location = 1) out vec4 normal_world;
layout(location = 2) out vec4 v;
layout(location = 3) out vec4 v_world;
layout(location = 4) out vec2 uv;
layout(location = 5) out mat3 TBN;
layout(location = 8) out vec3 v_tangent;
layout(location = 9) out vec3 camPos_tangent;

////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void main(void)
{
	// Calculate the position of the vertex against the world, view, and projection matrices.
	v_world = vec4(inputPosition, 1.0f);
	v = viewMatrix * v_world;
	gl_Position = projectionMatrix * v;

    normal_world = vec4(0.0f, 0.0f, 1.0f, 0.0f);
    normal = normalize(viewMatrix * normal_world);

    vec3 tangent = vec3(1.0f, 0.0f, 0.0f);
    vec3 bitangent = vec3(0.0f, 1.0f, 0.0f);

    // re-orthogonalize T with respect to N
    tangent = normalize(tangent - dot(tangent, normal_world.xyz) * normal_world.xyz);
    // then retrieve perpendicular vector B with the cross product of T and N
    bitangent = cross(normal_world.xyz, tangent);

    TBN = mat3(tangent, bitangent, normal_world.xyz);
    mat3 TBN_trans = transpose(TBN);

    v_tangent = TBN_trans * v_world.xyz;
    camPos_tangent = TBN_trans * camPos.xyz;

    uv.x = inputUV.x;
    uv.y = 1.0f - inputUV.y;
}

