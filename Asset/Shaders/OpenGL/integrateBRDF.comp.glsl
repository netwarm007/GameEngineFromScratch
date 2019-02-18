#version 420
#extension GL_ARB_compute_shader : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Light
{
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    vec2 lightSize;
    ivec4 lightGuid;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(binding = 0, rg32f) uniform writeonly image2D img_output;

float RadicalInverse_VdC(inout uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 1431655765u) << 1u) | ((bits & 2863311530u) >> 1u);
    bits = ((bits & 858993459u) << 2u) | ((bits & 3435973836u) >> 2u);
    bits = ((bits & 252645135u) << 4u) | ((bits & 4042322160u) >> 4u);
    bits = ((bits & 16711935u) << 8u) | ((bits & 4278255360u) >> 8u);
    return float(bits) * 2.3283064365386962890625e-10;
}

vec2 Hammersley(uint i, uint N)
{
    uint param = i;
    float _162 = RadicalInverse_VdC(param);
    return vec2(float(i) / float(N), _162);
}

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
    float a = roughness * roughness;
    float phi = 6.283185482025146484375 * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (((a * a) - 1.0) * Xi.y)));
    float sinTheta = sqrt(1.0 - (cosTheta * cosTheta));
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
    bvec3 _219 = bvec3(abs(N.z) < 0.999000012874603271484375);
    vec3 up = vec3(_219.x ? vec3(0.0, 0.0, 1.0).x : vec3(1.0, 0.0, 0.0).x, _219.y ? vec3(0.0, 0.0, 1.0).y : vec3(1.0, 0.0, 0.0).y, _219.z ? vec3(0.0, 0.0, 1.0).z : vec3(1.0, 0.0, 0.0).z);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    vec3 sampleVec = ((tangent * H.x) + (bitangent * H.y)) + (N * H.z);
    return normalize(sampleVec);
}

float GeometrySchlickGGXIndirect(float NdotV, float roughness)
{
    float a = roughness;
    float k = (a * a) / 2.0;
    float nom = NdotV;
    float denom = (NdotV * (1.0 - k)) + k;
    return nom / denom;
}

float GeometrySmithIndirect(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float param = NdotV;
    float param_1 = roughness;
    float ggx2 = GeometrySchlickGGXIndirect(param, param_1);
    float param_2 = NdotL;
    float param_3 = roughness;
    float ggx1 = GeometrySchlickGGXIndirect(param_2, param_3);
    return ggx1 * ggx2;
}

vec2 IntegrateBRDF(float NdotV, float roughness)
{
    vec3 V;
    V.x = sqrt(1.0 - (NdotV * NdotV));
    V.y = 0.0;
    V.z = NdotV;
    float A = 0.0;
    float B = 0.0;
    vec3 N = vec3(0.0, 0.0, 1.0);
    for (uint i = 0u; i < 1024u; i++)
    {
        uint param = i;
        uint param_1 = 1024u;
        vec2 Xi = Hammersley(param, param_1);
        vec2 param_2 = Xi;
        vec3 param_3 = N;
        float param_4 = roughness;
        vec3 H = ImportanceSampleGGX(param_2, param_3, param_4);
        vec3 L = normalize((H * (2.0 * dot(V, H))) - V);
        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);
        if (NdotL > 0.0)
        {
            vec3 param_5 = N;
            vec3 param_6 = V;
            vec3 param_7 = L;
            float param_8 = roughness;
            float G = GeometrySmithIndirect(param_5, param_6, param_7, param_8);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);
            A += ((1.0 - Fc) * G_Vis);
            B += (Fc * G_Vis);
        }
    }
    A /= 1024.0;
    B /= 1024.0;
    return vec2(A, B);
}

void _integrateBRDF_comp_main(uvec3 DTid)
{
    ivec2 pixel_coords = ivec2(DTid.xy);
    float param = float(pixel_coords.x + 1) / 512.0;
    float param_1 = float(pixel_coords.y + 1) / 512.0;
    vec2 _383 = IntegrateBRDF(param, param_1);
    vec4 pixel;
    pixel = vec4(_383.x, _383.y, pixel.z, pixel.w);
    vec2 storeTemp = pixel.xy;
    imageStore(img_output, pixel_coords, storeTemp.xyyy);
}

void main()
{
    uvec3 DTid = gl_GlobalInvocationID;
    uvec3 param = DTid;
    _integrateBRDF_comp_main(param);
}

