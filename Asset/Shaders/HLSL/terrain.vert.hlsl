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

Texture2D<float4> terrainHeightMap : register(t11, space0);
SamplerState _terrainHeightMap_sampler : register(s11, space0);
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

static float4 gl_Position;
static float3 inputPosition;

struct SPIRV_Cross_Input
{
    float3 inputPosition : TEXCOORD0;
};

struct SPIRV_Cross_Output
{
    float4 gl_Position : SV_Position;
};

void vert_main()
{
    float height = terrainHeightMap.SampleLevel(_terrainHeightMap_sampler, inputPosition.xy, 0.0f).x;
    float4 displaced = float4(inputPosition.xy, height, 1.0f);
    gl_Position = displaced;
}

SPIRV_Cross_Output terrain_vert_main(SPIRV_Cross_Input stage_input)
{
    inputPosition = stage_input.inputPosition;
    vert_main();
    SPIRV_Cross_Output stage_output;
    stage_output.gl_Position = gl_Position;
    return stage_output;
}
