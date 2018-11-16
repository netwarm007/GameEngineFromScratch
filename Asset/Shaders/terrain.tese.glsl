layout(quads, fractional_odd_spacing, ccw) in;

//////////////////////
// OUTPUT VARIABLES //
//////////////////////
layout(location = 1) out vec4 normal_world;
layout(location = 3) out vec4 v_world;
layout(location = 4) out vec2 uv;
layout(location = 5) out mat3 TBN;
layout(location = 8) out vec3 v_tangent;
layout(location = 9) out vec3 camPos_tangent;

void main(){
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    vec4 a = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, u);
    vec4 b = mix(gl_in[3].gl_Position, gl_in[2].gl_Position, u);
    v_world = mix(a, b, v);
    normal_world = vec4(0.0f, 0.0f, 1.0f, 0.0f);
    uv = gl_TessCoord.xy;
    float height = texture(terrainHeightMap, uv).r;
    gl_Position = projectionMatrix * viewMatrix * vec4(v_world.xy, height, 1.0);

    vec3 tangent = vec3(1.0f, 0.0f, 0.0f);
    vec3 bitangent = vec3(0.0f, 1.0f, 0.0f);

    TBN = mat3(tangent, bitangent, normal_world.xyz);
    mat3 TBN_trans = transpose(TBN);

    v_tangent = TBN_trans * v_world.xyz;
    camPos_tangent = TBN_trans * camPos.xyz;
}
