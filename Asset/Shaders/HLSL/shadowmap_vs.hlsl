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

cbuffer _25 : register(b1)
{
    row_major float4x4 _25_modelMatrix : packoffset(c0);
};
cbuffer u_pushConstants
{
    row_major float4x4 u_pushConstants_depthVP : packoffset(c0);
};
uniform float4 gl_HalfPixel;
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

static float4 gl_Position;
static float3 inputPosition;

struct SPIRV_Cross_Input
{
    float3 inputPosition : TEXCOORD0;
};

struct SPIRV_Cross_Output
{
    float4 gl_Position : POSITION;
};

void vert_main()
{
    gl_Position = mul(float4(inputPosition, 1.0f), mul(_25_modelMatrix, u_pushConstants_depthVP));
    gl_Position.x = gl_Position.x - gl_HalfPixel.x * gl_Position.w;
    gl_Position.y = gl_Position.y + gl_HalfPixel.y * gl_Position.w;
}

SPIRV_Cross_Output main(SPIRV_Cross_Input stage_input)
{
    inputPosition = stage_input.inputPosition;
    vert_main();
    SPIRV_Cross_Output stage_output;
    stage_output.gl_Position = gl_Position;
    return stage_output;
}
