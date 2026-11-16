TextureCubeArray skyboxMap : register(t0);
SamplerState skyboxSampler : register(s5);

cbuffer PerFrameConstants : register(b10)
{
    row_major float4x4 viewMatrix;
    row_major float4x4 projectionMatrix;
    float4 camPos;
    int numLights;
    int output_is_srgb;
};

float4 vulkan_skybox_frag_main(float3 direction : TEXCOORD0) : SV_Target
{
    float3 color =
        skyboxMap.SampleLevel(skyboxSampler, float4(direction, 0.0f), 0.0f).rgb;
    color = 1.0f - exp(-color);
    if (output_is_srgb == 0) {
        color = pow(max(color, 0.0f.xxx), 1.0f / 2.2f);
    }
    return float4(color, 1.0f);
}
