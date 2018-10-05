////////////////////////////////////////////////////////////////////////////////
// Filename: basic.vs
////////////////////////////////////////////////////////////////////////////////

/////////////////////
// INPUT VARIABLES //
/////////////////////
layout(location = 0) in vec3 inputPosition;
layout(location = 1) in vec3 inputNormal;
layout(location = 2) in vec2 inputUV;
layout(location = 3) in vec3 inputTangent;
layout(location = 4) in vec3 inputBiTangent;

//////////////////////
// OUTPUT VARIABLES //
//////////////////////
layout(location = 0) out vec4 normal;
layout(location = 1) out vec4 normal_world;
layout(location = 2) out vec4 v;
layout(location = 3) out vec4 v_world;
layout(location = 4) out vec2 uv;
layout(location = 5) out mat3 TBN;

////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void main(void)
{
	// Calculate the position of the vertex against the world, view, and projection matrices.
	v_world = modelMatrix * vec4(inputPosition, 1.0f);
	v = viewMatrix * v_world;
	gl_Position = projectionMatrix * v;

    normal_world = normalize(modelMatrix * vec4(inputNormal, 0.0f));
    normal = normalize(viewMatrix * normal_world);
    vec3 tangent = normalize(vec3(modelMatrix * vec4(inputTangent, 0.0f)));
    //vec3 bitangent = normalize(vec3(modelMatrix * vec4(inputBiTangent, 0.0f)));
    // re-orthogonalize T with respect to N
    tangent = normalize(tangent - dot(tangent, normal_world.xyz) * normal_world.xyz);
    // then retrieve perpendicular vector B with the cross product of T and N
    vec3 bitangent = cross(normal_world.xyz, tangent);

    TBN = mat3(tangent, bitangent, normal_world.xyz);

    uv.x = inputUV.x;
    uv.y = 1.0f - inputUV.y;
}

