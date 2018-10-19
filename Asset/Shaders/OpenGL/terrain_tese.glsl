#version 400
layout(quads, ccw, fractional_odd_spacing) in;

struct Light
{
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    vec2 lightSize;
    ivec4 lightGUID;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
    Light allLights[100];
} _90;

uniform sampler2D terrainHeightMap;

out vec4 v_world;
out vec4 normal_world;
out vec2 uv;
out mat3 TBN;
out vec3 v_tangent;
out vec3 camPos_tangent;

void main()
{
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;
    vec4 a = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, vec4(u));
    vec4 b = mix(gl_in[3].gl_Position, gl_in[2].gl_Position, vec4(u));
    v_world = mix(a, b, vec4(v));
    normal_world = vec4(0.0, 0.0, 1.0, 0.0);
    uv = gl_TessCoord.xy;
    float height = textureLod(terrainHeightMap, uv, 0.0).x;
    gl_Position = (_90.projectionMatrix * _90.viewMatrix) * vec4(v_world.xy, height, 1.0);
    vec3 tangent = vec3(1.0, 0.0, 0.0);
    vec3 bitangent = vec3(0.0, 1.0, 0.0);
    TBN = mat3(vec3(tangent), vec3(bitangent), vec3(normal_world.xyz));
    mat3 TBN_trans = transpose(TBN);
    v_tangent = TBN_trans * v_world.xyz;
    camPos_tangent = TBN_trans * _90.camPos.xyz;
}

