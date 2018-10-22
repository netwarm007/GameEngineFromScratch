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
    row_major float4x4 lightVP;
    float4 padding[2];
};

cbuffer _13 : register(b1, space0)
{
    row_major float4x4 _13_modelMatrix : packoffset(c0);
};
cbuffer _42 : register(b0, space0)
{
    row_major float4x4 _42_viewMatrix : packoffset(c0);
    row_major float4x4 _42_projectionMatrix : packoffset(c4);
    float4 _42_camPos : packoffset(c8);
    int _42_numLights : packoffset(c9);
    Light _42_allLights[100] : packoffset(c10);
};
Texture2D<float4> diffuseMap : register(t0, space0);
SamplerState _diffuseMap_sampler : register(s0, space0);
Texture2DArray<float4> shadowMap : register(t1, space0);
SamplerState _shadowMap_sampler : register(s1, space0);
Texture2DArray<float4> globalShadowMap : register(t2, space0);
SamplerState _globalShadowMap_sampler : register(s2, space0);
TextureCubeArray<float4> cubeShadowMap : register(t3, space0);
SamplerState _cubeShadowMap_sampler : register(s3, space0);
TextureCubeArray<float4> skybox : register(t4, space0);
SamplerState _skybox_sampler : register(s4, space0);
Texture2D<float4> normalMap : register(t5, space0);
SamplerState _normalMap_sampler : register(s5, space0);
Texture2D<float4> metallicMap : register(t6, space0);
SamplerState _metallicMap_sampler : register(s6, space0);
Texture2D<float4> roughnessMap : register(t7, space0);
SamplerState _roughnessMap_sampler : register(s7, space0);
Texture2D<float4> aoMap : register(t8, space0);
SamplerState _aoMap_sampler : register(s8, space0);
Texture2D<float4> brdfLUT : register(t9, space0);
SamplerState _brdfLUT_sampler : register(s9, space0);
Texture2D<float4> heightMap : register(t10, space0);
SamplerState _heightMap_sampler : register(s10, space0);
Texture2D<float4> terrainHeightMap : register(t11, space0);
SamplerState _terrainHeightMap_sampler : register(s11, space0);

static float4 gl_Position;
static float4 v_world;
static float3 inputPosition;
static float4 v;
static float4 normal_world;
static float3 inputNormal;
static float4 normal;
static float3 inputTangent;
static float3x3 TBN;
static float3 v_tangent;
static float3 camPos_tangent;
static float2 uv;
static float2 inputUV;
static float3 inputBiTangent;

struct SPIRV_Cross_Input
{
    float3 inputPosition : TEXCOORD0;
    float3 inputNormal : TEXCOORD1;
    float2 inputUV : TEXCOORD2;
    float3 inputTangent : TEXCOORD3;
    float3 inputBiTangent : TEXCOORD4;
};

struct SPIRV_Cross_Output
{
    float4 normal : TEXCOORD0;
    float4 normal_world : TEXCOORD1;
    float4 v : TEXCOORD2;
    float4 v_world : TEXCOORD3;
    float2 uv : TEXCOORD4;
    float3x3 TBN : TEXCOORD5;
    float3 v_tangent : TEXCOORD8;
    float3 camPos_tangent : TEXCOORD9;
    float4 gl_Position : SV_Position;
};

void vert_main()
{
    v_world = mul(float4(inputPosition, 1.0f), _13_modelMatrix);
    v = mul(v_world, _42_viewMatrix);
    gl_Position = mul(v, _42_projectionMatrix);
    normal_world = normalize(mul(float4(inputNormal, 0.0f), _13_modelMatrix));
    normal = normalize(mul(normal_world, _42_viewMatrix));
    float3 tangent = normalize(float3(mul(float4(inputTangent, 0.0f), _13_modelMatrix).xyz));
    tangent = normalize(tangent - (normal_world.xyz * dot(tangent, normal_world.xyz)));
    float3 bitangent = cross(normal_world.xyz, tangent);
    TBN = float3x3(float3(tangent), float3(bitangent), float3(normal_world.xyz));
    float3x3 TBN_trans = transpose(TBN);
    v_tangent = mul(v_world.xyz, TBN_trans);
    camPos_tangent = mul(_42_camPos.xyz, TBN_trans);
    uv.x = inputUV.x;
    uv.y = 1.0f - inputUV.y;
}

SPIRV_Cross_Output main(SPIRV_Cross_Input stage_input)
{
    inputPosition = stage_input.inputPosition;
    inputNormal = stage_input.inputNormal;
    inputTangent = stage_input.inputTangent;
    inputUV = stage_input.inputUV;
    inputBiTangent = stage_input.inputBiTangent;
    vert_main();
    SPIRV_Cross_Output stage_output;
    stage_output.gl_Position = gl_Position;
    stage_output.v_world = v_world;
    stage_output.v = v;
    stage_output.normal_world = normal_world;
    stage_output.normal = normal;
    stage_output.TBN = TBN;
    stage_output.v_tangent = v_tangent;
    stage_output.camPos_tangent = camPos_tangent;
    stage_output.uv = uv;
    return stage_output;
}
