#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 arbitraryMatrix;
    float4 camPos;
    int numLights;
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

struct Light
{
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    float2 lightSize;
    int4 lightGuid;
    float4 lightPosition;
    float4 lightColor;
    float4 lightDirection;
    float4 lightDistAttenCurveParams[2];
    float4 lightAngleAttenCurveParams[2];
    float4x4 lightVP;
    float4 padding[2];
};

struct LightInfo
{
    Light lights[100];
};

float RadicalInverse_VdC(thread uint& bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 1431655765u) << 1u) | ((bits & 2863311530u) >> 1u);
    bits = ((bits & 858993459u) << 2u) | ((bits & 3435973836u) >> 2u);
    bits = ((bits & 252645135u) << 4u) | ((bits & 4042322160u) >> 4u);
    bits = ((bits & 16711935u) << 8u) | ((bits & 4278255360u) >> 8u);
    return float(bits) * 2.3283064365386962890625e-10;
}

float2 Hammersley(thread const uint& i, thread const uint& N)
{
    uint param = i;
    float _162 = RadicalInverse_VdC(param);
    return float2(float(i) / float(N), _162);
}

float3 ImportanceSampleGGX(thread const float2& Xi, thread const float3& N, thread const float& roughness)
{
    float a = roughness * roughness;
    float phi = 6.283185482025146484375 * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (((a * a) - 1.0) * Xi.y)));
    float sinTheta = sqrt(1.0 - (cosTheta * cosTheta));
    float3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
    bool3 _219 = bool3(abs(N.z) < 0.999000012874603271484375);
    float3 up = float3(_219.x ? float3(0.0, 0.0, 1.0).x : float3(1.0, 0.0, 0.0).x, _219.y ? float3(0.0, 0.0, 1.0).y : float3(1.0, 0.0, 0.0).y, _219.z ? float3(0.0, 0.0, 1.0).z : float3(1.0, 0.0, 0.0).z);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);
    float3 sampleVec = ((tangent * H.x) + (bitangent * H.y)) + (N * H.z);
    return normalize(sampleVec);
}

float GeometrySchlickGGXIndirect(thread const float& NdotV, thread const float& roughness)
{
    float a = roughness;
    float k = (a * a) / 2.0;
    float nom = NdotV;
    float denom = (NdotV * (1.0 - k)) + k;
    return nom / denom;
}

float GeometrySmithIndirect(thread const float3& N, thread const float3& V, thread const float3& L, thread const float& roughness)
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

float2 IntegrateBRDF(thread const float& NdotV, thread const float& roughness)
{
    float3 V;
    V.x = sqrt(1.0 - (NdotV * NdotV));
    V.y = 0.0;
    V.z = NdotV;
    float A = 0.0;
    float B = 0.0;
    float3 N = float3(0.0, 0.0, 1.0);
    for (uint i = 0u; i < 1024u; i++)
    {
        uint param = i;
        uint param_1 = 1024u;
        float2 Xi = Hammersley(param, param_1);
        float2 param_2 = Xi;
        float3 param_3 = N;
        float param_4 = roughness;
        float3 H = ImportanceSampleGGX(param_2, param_3, param_4);
        float3 L = normalize((H * (2.0 * dot(V, H))) - V);
        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);
        if (NdotL > 0.0)
        {
            float3 param_5 = N;
            float3 param_6 = V;
            float3 param_7 = L;
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
    return float2(A, B);
}

void _integrateBRDF_comp_main(thread const uint3& DTid, thread texture2d<float, access::write> img_output)
{
    int2 pixel_coords = int2(DTid.xy);
    float param = float(pixel_coords.x + 1) / 512.0;
    float param_1 = float(pixel_coords.y + 1) / 512.0;
    float2 _383 = IntegrateBRDF(param, param_1);
    float4 pixel;
    pixel = float4(_383.x, _383.y, pixel.z, pixel.w);
    img_output.write(pixel, uint2(pixel_coords));
}

kernel void integrateBRDF_comp_main(texture2d<float, access::write> img_output [[texture(0)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    uint3 DTid = gl_GlobalInvocationID;
    uint3 param = DTid;
    _integrateBRDF_comp_main(param, img_output);
}

