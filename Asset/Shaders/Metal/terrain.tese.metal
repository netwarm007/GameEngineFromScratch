#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Light
{
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    float2 lightSize;
    int4 lightGUID;
    float4 lightPosition;
    float4 lightColor;
    float4 lightDirection;
    float4 lightDistAttenCurveParams[2];
    float4 lightAngleAttenCurveParams[2];
    float4x4 lightVP;
    float4 padding[2];
};

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4 camPos;
    int numLights;
    Light allLights[100];
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

struct terrain_tese_main_out
{
    float4 normal_world;
    float4 v_world;
    float2 uv;
    float3 TBN_0;
    float3 TBN_1;
    float3 TBN_2;
    float3 v_tangent;
    float3 camPos_tangent;
    float4 gl_Position;
};

unknown terrain_tese_main_out terrain_tese_main(constant PerFrameConstants& _90 [[buffer(1)]], texture2d<float> terrainHeightMap [[texture(11)]], sampler terrainHeightMapSmplr [[sampler(11)]], unsupported-built-in-type gl_TessCoord [[unsupported-built-in]], float4 gl_in [[position]])
{
    terrain_tese_main_out out = {};
    float3x3 TBN = {};
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;
    float4 a = mix(gl_in[0].out.gl_Position, gl_in[1].out.gl_Position, float4(u));
    float4 b = mix(gl_in[3].out.gl_Position, gl_in[2].out.gl_Position, float4(u));
    out.v_world = mix(a, b, float4(v));
    out.normal_world = float4(0.0, 0.0, 1.0, 0.0);
    out.uv = gl_TessCoord.xy;
    float height = terrainHeightMap.sample(terrainHeightMapSmplr, out.uv, level(0.0)).x;
    out.gl_Position = (_90.projectionMatrix * _90.viewMatrix) * float4(out.v_world.xy, height, 1.0);
    float3 tangent = float3(1.0, 0.0, 0.0);
    float3 bitangent = float3(0.0, 1.0, 0.0);
    TBN = float3x3(float3(tangent), float3(bitangent), float3(out.normal_world.xyz));
    float3x3 TBN_trans = transpose(TBN);
    out.v_tangent = TBN_trans * out.v_world.xyz;
    out.camPos_tangent = TBN_trans * _90.camPos.xyz;
    out.TBN_0 = TBN[0];
    out.TBN_1 = TBN[1];
    out.TBN_2 = TBN[2];
    return out;
}

