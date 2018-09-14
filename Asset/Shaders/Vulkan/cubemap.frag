#version 450

/////////////////////
// CONSTANTS       //
/////////////////////
// per frame
#define MAX_LIGHTS 100

struct Light {
    int  lightType;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightSize;
    float lightIntensity;
    mat4 lightDistAttenCurveParams;
    mat4 lightAngleAttenCurveParams;
    mat4 lightVP;
    int  lightShadowMapIndex;
};

layout(std140,binding=0) uniform DrawFrameConstants {
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec3 ambientColor;
    vec3 camPos;
    int numLights;
    Light allLights[MAX_LIGHTS];
};

// per drawcall
layout(std140,binding=1) uniform DrawBatchConstants {
    mat4 modelMatrix;

    vec3 diffuseColor;
    vec3 specularColor;
    float specularPower;
    float metallic;
    float roughness;
    float ao;

    bool usingDiffuseMap;
    bool usingNormalMap;
    bool usingMetallicMap;
    bool usingRoughnessMap;
    bool usingAoMap;
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
#define PI 3.14159265359

vec3 projectOnPlane(vec3 point, vec3 center_of_plane, vec3 normal_of_plane)
{
    return point - dot(point - center_of_plane, normal_of_plane) * normal_of_plane;
}

bool isAbovePlane(vec3 point, vec3 center_of_plane, vec3 normal_of_plane)
{
    return dot(point - center_of_plane, normal_of_plane) > 0.0f;
}

vec3 linePlaneIntersect(vec3 line_start, vec3 line_dir, vec3 center_of_plane, vec3 normal_of_plane)
{
    return line_start + line_dir * (dot(center_of_plane - line_start, normal_of_plane) / dot(line_dir, normal_of_plane));
}

float linear_interpolate(float t, float begin, float end)
{
    if (t < begin)
    {
        return 1.0f;
    }
    else if (t > end)
    {
        return 0.0f;
    }
    else
    {
        return (end - t) / (end - begin);
    }
}

float apply_atten_curve(float dist, mat4 atten_params)
{
    float atten = 1.0f;

    switch(int(atten_params[0][0]))
    {
        case 1: // linear
        {
            float begin_atten = atten_params[0][1];
            float end_atten = atten_params[0][2];
            atten = linear_interpolate(dist, begin_atten, end_atten);
            break;
        }
        case 2: // smooth
        {
            float begin_atten = atten_params[0][1];
            float end_atten = atten_params[0][2];
            float tmp = linear_interpolate(dist, begin_atten, end_atten);
            atten = 3.0f * pow(tmp, 2.0f) - 2.0f * pow(tmp, 3.0f);
            break;
        }
        case 3: // inverse
        {
            float scale = atten_params[0][1];
            float offset = atten_params[0][2];
            float kl = atten_params[0][3];
            float kc = atten_params[1][0];
            atten = clamp(scale / 
                (kl * dist + kc * scale) + offset, 
                0.0f, 1.0f);
            break;
        }
        case 4: // inverse square
        {
            float scale = atten_params[0][1];
            float offset = atten_params[0][2];
            float kq = atten_params[0][3];
            float kl = atten_params[1][0];
            float kc = atten_params[1][1];
            atten = clamp(pow(scale, 2.0f) / 
                (kq * pow(dist, 2.0f) + kl * dist * scale + kc * pow(scale, 2.0f) + offset), 
                0.0f, 1.0f);
            break;
        }
        case 0:
        default:
            break; // no attenuation
    }

    return atten;
}

float shadow_test(const vec4 p, const Light light, const float cosTheta) {
    vec4 v_light_space = light.lightVP * p;
    v_light_space /= v_light_space.w;

    const mat4 depth_bias = mat4 (
        vec4(0.5f, 0.0f, 0.0f, 0.0f),
        vec4(0.0f, 0.5f, 0.0f, 0.0f),
        vec4(0.0f, 0.0f, 0.5f, 0.0f),
        vec4(0.5f, 0.5f, 0.5f, 1.0f)
    );

    const vec2 poissonDisk[4] = vec2[](
        vec2( -0.94201624f, -0.39906216f ),
        vec2( 0.94558609f, -0.76890725f ),
        vec2( -0.094184101f, -0.92938870f ),
        vec2( 0.34495938f, 0.29387760f )
    );

    // shadow test
    float visibility = 1.0f;
    if (light.lightShadowMapIndex != -1) // the light cast shadow
    {
        float bias = (5e-4) * tan(acos(cosTheta)); // cosTheta is dot( n,l ), clamped between 0 and 1
        bias = clamp(bias, 0.0f, 0.01f);
        float near_occ;
        switch (light.lightType)
        {
            case 0: // point
                // recalculate the v_light_space because we do not need to taking account of rotation
                vec3 L = p.xyz - light.lightPosition.xyz;
                near_occ = texture(cubeShadowMap, vec4(L, light.lightShadowMapIndex)).r;

                if (length(L) - near_occ * 10.0f > bias)
                {
                    // we are in the shadow
                    visibility -= 0.88f;
                }
                break;
            case 1: // spot
                // adjust from [-1, 1] to [0, 1]
                v_light_space = depth_bias * v_light_space;
                for (int i = 0; i < 4; i++)
                {
                    near_occ = texture(shadowMap, vec3(v_light_space.xy + poissonDisk[i] / 700.0f, light.lightShadowMapIndex)).r;

                    if (v_light_space.z - near_occ > bias)
                    {
                        // we are in the shadow
                        visibility -= 0.22f;
                    }
                }
                break;
            case 2: // infinity
                // adjust from [-1, 1] to [0, 1]
                v_light_space = depth_bias * v_light_space;
                for (int i = 0; i < 4; i++)
                {
                    near_occ = texture(globalShadowMap, vec3(v_light_space.xy + poissonDisk[i] / 700.0f, light.lightShadowMapIndex)).r;

                    if (v_light_space.z - near_occ > bias)
                    {
                        // we are in the shadow
                        visibility -= 0.22f;
                    }
                }
                break;
            case 3: // area
                // adjust from [-1, 1] to [0, 1]
                v_light_space = depth_bias * v_light_space;
                for (int i = 0; i < 4; i++)
                {
                    near_occ = texture(shadowMap, vec3(v_light_space.xy + poissonDisk[i] / 700.0f, light.lightShadowMapIndex)).r;

                    if (v_light_space.z - near_occ > bias)
                    {
                        // we are in the shadow
                        visibility -= 0.22f;
                    }
                }
                break;
        }
    }

    return visibility;
}

vec3 reinhard_tone_mapping(vec3 color)
{
    return color / (color + vec3(1.0f));
}

vec3 exposure_tone_mapping(vec3 color)
{
    const float exposure = 1.0f;
    return vec3(1.0f) - exp(-color * exposure);
}

vec3 gamma_correction(vec3 color)
{
    const float gamma = 2.2f;
    return pow(color, vec3(1.0f / gamma));
}

vec3 inverse_gamma_correction(vec3 color)
{
    const float gamma = 2.2f;
    return pow(color, vec3(gamma));
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0f - roughness), F0) - F0) * pow(1.0f - cosTheta, 5.0f);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0f);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;
	
    return num / denom;
}

float GeometrySchlickGGXDirect(float NdotV, float roughness)
{
    float r = (roughness + 1.0f);
    float k = (r*r) / 8.0f;

    float num   = NdotV;
    float denom = NdotV * (1.0f - k) + k;
	
    return num / denom;
}

float GeometrySmithDirect(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx2  = GeometrySchlickGGXDirect(NdotV, roughness);
    float ggx1  = GeometrySchlickGGXDirect(NdotL, roughness);
	
    return ggx1 * ggx2;
}

float GeometrySchlickGGXIndirect(float NdotV, float roughness)
{
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmithIndirect(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGXIndirect(NdotV, roughness);
    float ggx1 = GeometrySchlickGGXIndirect(NdotL, roughness);

    return ggx1 * ggx2;
}

float RadicalInverse_VdC(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}  

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
    float a = roughness*roughness;
	
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    // from spherical coordinates to cartesian coordinates
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
	
    // from tangent-space vector to world-space sample vector
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
	
    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}
layout(location = 0) in vec3 UVW;

layout(location = 0) out vec3 color;

layout(push_constant) uniform debugPushConstants {
    float level;
} u_pushConstants;

layout(binding = 0) uniform samplerCube depthSampler;

void main(){
    color = textureLod(depthSampler, UVW, u_pushConstants.level).rgb;
}
