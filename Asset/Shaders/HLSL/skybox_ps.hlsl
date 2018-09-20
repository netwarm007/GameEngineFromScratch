struct Light
{
    int lightType;
    float lightIntensity;
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

uniform samplerCUBEArray skybox;
uniform sampler2D diffuseMap;
uniform sampler2DArray shadowMap;
uniform sampler2DArray globalShadowMap;
uniform samplerCUBEArray cubeShadowMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform sampler2D brdfLUT;

static float4 outputColor;
static float3 UVW;

struct SPIRV_Cross_Input
{
    float3 UVW : TEXCOORD0;
};

struct SPIRV_Cross_Output
{
    float4 outputColor : COLOR0;
};

float3 inverse_gamma_correction(float3 color)
{
    return pow(color, 2.2000000476837158203125f.xxx);
}

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
    outputColor = texCUBElod(skybox, float4(float4(UVW, 0.0f), 0.0f));
    float3 param = outputColor.xyz;
    float3 _60 = inverse_gamma_correction(param);
    outputColor = float4(_60.x, _60.y, _60.z, outputColor.w);
    float3 param_1 = outputColor.xyz;
    float3 _66 = exposure_tone_mapping(param_1);
    outputColor = float4(_66.x, _66.y, _66.z, outputColor.w);
    float3 param_2 = outputColor.xyz;
    float3 _72 = gamma_correction(param_2);
    outputColor = float4(_72.x, _72.y, _72.z, outputColor.w);
}

SPIRV_Cross_Output main(SPIRV_Cross_Input stage_input)
{
    UVW = stage_input.UVW;
    frag_main();
    SPIRV_Cross_Output stage_output;
    stage_output.outputColor = outputColor;
    return stage_output;
}
