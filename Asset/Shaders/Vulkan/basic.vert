#version 450
/////////////
// consts

#define MAX_LIGHTS 100
#define PI 3.14159265359
/////////////////////
// CONSTANTS       //
/////////////////////
// per frame
struct Light {
    float   lightIntensity;
    int     lightType;
    int     lightCastShadow;
    int     lightShadowMapIndex;
    int     lightAngleAttenCurveType;
    int     lightDistAttenCurveType;
    vec2    lightSize;
    ivec4   lightGUID;
    vec4    lightPosition;
    vec4    lightColor;
    vec4    lightDirection;
    vec4    lightDistAttenCurveParams[2];
    vec4    lightAngleAttenCurveParams[2];
    mat4    lightVP;
    vec4    padding[2];
};

layout(std140,binding=0) uniform PerFrameConstants {
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
    Light allLights[MAX_LIGHTS];
};

// per drawcall
layout(std140,binding=1) uniform PerBatchConstants {
    mat4 modelMatrix;
};

// samplers
layout(binding = 0) uniform sampler2D diffuseMap;
layout(binding = 1) uniform sampler2DArray shadowMap;
layout(binding = 2) uniform sampler2DArray globalShadowMap;
layout(binding = 3) uniform samplerCubeArray cubeShadowMap;
layout(binding = 4) uniform samplerCubeArray skybox;
layout(binding = 5) uniform sampler2D normalMap;
layout(binding = 6) uniform sampler2D metallicMap;
layout(binding = 7) uniform sampler2D roughnessMap;
layout(binding = 8) uniform sampler2D aoMap;
layout(binding = 9) uniform sampler2D brdfLUT;
layout(binding = 10) uniform sampler2D heightMap;
layout(binding = 11) uniform sampler2D terrainHeightMap;

vec3 projectOnPlane(vec3 point, vec3 center_of_plane, vec3 normal_of_plane);
bool isAbovePlane(vec3 point, vec3 center_of_plane, vec3 normal_of_plane);
vec3 linePlaneIntersect(vec3 line_start, vec3 line_dir, vec3 center_of_plane, vec3 normal_of_plane);
float linear_interpolate(float t, float begin, float end);
float apply_atten_curve(float dist, int atten_curve_type, vec4 atten_params[2]);
float shadow_test(const vec4 p, const Light light, const float cosTheta) ;
vec3 reinhard_tone_mapping(vec3 color);
vec3 exposure_tone_mapping(vec3 color);
vec3 gamma_correction(vec3 color);
vec3 inverse_gamma_correction(vec3 color);
vec3 fresnelSchlick(float cosTheta, vec3 F0);
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness);
float DistributionGGX(vec3 N, vec3 H, float roughness);
float GeometrySchlickGGXDirect(float NdotV, float roughness);
float GeometrySmithDirect(vec3 N, vec3 V, vec3 L, float roughness);
float GeometrySchlickGGXIndirect(float NdotV, float roughness);
float GeometrySmithIndirect(vec3 N, vec3 V, vec3 L, float roughness);
float RadicalInverse_VdC(uint bits) ;
vec2 Hammersley(uint i, uint N);
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness);
vec2 ParallaxMapping(vec2 texCoords, vec3 viewDir);
vec4 project(vec4 vertex);
vec2 screen_space(vec4 vertex);
bool offscreen(vec4 vertex);
float level(vec2 v0, vec2 v1);
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
layout(location = 8) out vec3 v_tangent;
layout(location = 9) out vec3 camPos_tangent;

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
#if 0
    vec3 bitangent = normalize(vec3(modelMatrix * vec4(inputBiTangent, 0.0f)));
#endif
    // re-orthogonalize T with respect to N
    tangent = normalize(tangent - dot(tangent, normal_world.xyz) * normal_world.xyz);
    // then retrieve perpendicular vector B with the cross product of T and N
    vec3 bitangent = cross(normal_world.xyz, tangent);

    TBN = mat3(tangent, bitangent, normal_world.xyz);
    mat3 TBN_trans = transpose(TBN);

    v_tangent = TBN_trans * v_world.xyz;
    camPos_tangent = TBN_trans * camPos.xyz;

    uv.x = inputUV.x;
    uv.y = 1.0f - inputUV.y;
}

