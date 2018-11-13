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
layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

layout(push_constant) uniform gs_constant_t {
    float layer_index;
} u_gsPushConstants;

layout(std140,binding=2) uniform ShadowMatrices {
    mat4 shadowMatrices[6];
};

layout(location = 0) out vec4 FragPos; // FragPos from GS (output per emitvertex)

void main()
{
    for(int face = 0; face < 6; face++)
    {
        gl_Layer = int(u_gsPushConstants.layer_index) * 6 + face; // built-in variable that specifies to which face we render.
        for(int i = 0; i < 3; ++i) // for each triangle's vertices
        {
            FragPos = gl_in[i].gl_Position;
            gl_Position = shadowMatrices[face] * FragPos;
            EmitVertex();
        }    
        EndPrimitive();
    }
}  
