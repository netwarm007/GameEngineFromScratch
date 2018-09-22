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

cbuffer u_pushConstants
{
    float u_pushConstants_level : packoffset(c0);
    float u_pushConstants_layer_index : packoffset(c0.y);
};
uniform samplerCUBEArray depthSampler;
uniform sampler2D diffuseMap;
uniform sampler2DArray shadowMap;
uniform sampler2DArray globalShadowMap;
uniform samplerCUBEArray cubeShadowMap;
uniform samplerCUBEArray skybox;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform sampler2D brdfLUT;

static float3 color;
static float3 UVW;

struct SPIRV_Cross_Input
{
    float3 UVW : TEXCOORD0;
};

struct SPIRV_Cross_Output
{
    float3 color : COLOR0;
};

void frag_main()
{
    color = texCUBElod(depthSampler, float4(float4(UVW, u_pushConstants_layer_index), u_pushConstants_level)).xyz;
}

SPIRV_Cross_Output main(SPIRV_Cross_Input stage_input)
{
    UVW = stage_input.UVW;
    frag_main();
    SPIRV_Cross_Output stage_output;
    stage_output.color = color;
    return stage_output;
}
