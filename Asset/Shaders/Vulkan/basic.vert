#version 450

/////////////////////
// CONSTANTS       //
/////////////////////
// per frame
#define MAX_LIGHTS 100

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

float apply_atten_curve(float dist, int atten_curve_type, vec4 atten_params[2])
{
    float atten = 1.0f;

    switch(atten_curve_type)
    {
        case 1: // linear
        {
            float begin_atten = atten_params[0].x;
            float end_atten = atten_params[0].y;
            atten = linear_interpolate(dist, begin_atten, end_atten);
            break;
        }
        case 2: // smooth
        {
            float begin_atten = atten_params[0].x;
            float end_atten = atten_params[0].y;
            float tmp = linear_interpolate(dist, begin_atten, end_atten);
            atten = 3.0f * pow(tmp, 2.0f) - 2.0f * pow(tmp, 3.0f);
            break;
        }
        case 3: // inverse
        {
            float scale = atten_params[0].x;
            float offset = atten_params[0].y;
            float kl = atten_params[0].z;
            float kc = atten_params[0].w;
            atten = clamp(scale / 
                (kl * dist + kc * scale) + offset, 
                0.0f, 1.0f);
            break;
        }
        case 4: // inverse square
        {
            float scale = atten_params[0].x;
            float offset = atten_params[0].y;
            float kq = atten_params[0].z;
            float kl = atten_params[0].w;
            float kc = atten_params[1].x;
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

vec2 ParallaxMapping(vec2 texCoords, vec3 viewDir)
{ 
    const float height_scale = 0.10f;

    // number of depth layers
    const float minLayers = 8.0;
    const float maxLayers = 32.0;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), viewDir)));
    // calculate the size of each layer
    const float layerDepth = 1.0f / numLayers;
    // depth of current layer
    float currentLayerDepth = 0.0f;

    // get initial values
    vec2  currentTexCoords     = texCoords;
    float currentDepthMapValue = 1.0f - texture(heightMap, currentTexCoords).r;

    // the amount to shift the texture coordinates per layer (from vector P)
    const vec2 P = viewDir.xy * height_scale; 
    const vec2 deltaTexCoords = P / numLayers;
    
    while(currentLayerDepth < currentDepthMapValue)
    {
        // shift texture coordinates along direction of P
        currentTexCoords -= deltaTexCoords;
        // get depthmap value at current texture coordinates
        currentDepthMapValue = 1.0f - texture(heightMap, currentTexCoords).r;  
        // get depth of next layer
        currentLayerDepth += layerDepth;  
    }

    // get texture coordinates before collision (reverse operations)
    vec2 prevTexCoords = currentTexCoords + deltaTexCoords;

    // get depth after and before collision for linear interpolation
    float afterDepth  = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = 1.0f - texture(heightMap, prevTexCoords).r - currentLayerDepth + layerDepth;
    
    // interpolation of texture coordinates
    float weight = afterDepth / (afterDepth - beforeDepth);
    vec2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0f - weight);

    return finalTexCoords;  
} 
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

