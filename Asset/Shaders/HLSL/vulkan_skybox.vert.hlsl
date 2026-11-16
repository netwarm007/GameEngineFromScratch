cbuffer PerFrameConstants : register(b10)
{
    row_major float4x4 viewMatrix;
    row_major float4x4 projectionMatrix;
    float4 camPos;
    int numLights;
    int clip_space_type;
};

struct VertexInput
{
    float3 position : POSITION;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 direction : TEXCOORD0;
};

VertexOutput vulkan_skybox_vert_main(VertexInput input)
{
    VertexOutput output;
    float4x4 view = viewMatrix;
    view[3][0] = 0.0f;
    view[3][1] = 0.0f;
    view[3][2] = 0.0f;
    float4 clipPosition =
        mul(mul(float4(input.position, 1.0f), view), projectionMatrix);
    output.position = clipPosition.xyww;
    output.direction = input.position;
    return output;
}
