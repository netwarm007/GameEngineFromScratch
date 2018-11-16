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

TextureCubeArray<float4> skybox : register(t4, space0);
SamplerState _skybox_sampler : register(s4, space0);
Texture2D<float4> diffuseMap : register(t0, space0);
SamplerState _diffuseMap_sampler : register(s0, space0);
Texture2DArray<float4> shadowMap : register(t1, space0);
SamplerState _shadowMap_sampler : register(s1, space0);
Texture2DArray<float4> globalShadowMap : register(t2, space0);
SamplerState _globalShadowMap_sampler : register(s2, space0);
TextureCubeArray<float4> cubeShadowMap : register(t3, space0);
SamplerState _cubeShadowMap_sampler : register(s3, space0);
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

static float4 outputColor;
static float3 UVW;

struct SPIRV_Cross_Input
{
    float3 UVW : TEXCOORD0;
};

struct SPIRV_Cross_Output
{
    float4 outputColor : SV_Target0;
};

float3 exposure_tone_mapping(float3 color)
{
    return 1.0f.xxx - exp((-color) * 1.0f);
}

float3 gamma_correction(float3 color)
{
    return pow(color, 0.4545454680919647216796875f.xxx);
}

void frag_main()
{
    outputColor = skybox.SampleLevel(_skybox_sampler, float4(UVW, 0.0f), 0.0f);
    float3 param = outputColor.xyz;
    float3 _51 = exposure_tone_mapping(param);
    outputColor = float4(_51.x, _51.y, _51.z, outputColor.w);
    float3 param_1 = outputColor.xyz;
    float3 _57 = gamma_correction(param_1);
    outputColor = float4(_57.x, _57.y, _57.z, outputColor.w);
}

SPIRV_Cross_Output main(SPIRV_Cross_Input stage_input)
{
    UVW = stage_input.UVW;
    frag_main();
    SPIRV_Cross_Output stage_output;
    stage_output.outputColor = outputColor;
    return stage_output;
}
