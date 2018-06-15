#define SamplesMSAA 4

Texture2DMS<float4, SamplesMSAA> msaaTexture : register(t0);
  
cbuffer msaaDesc : register(b0)
{
	uint2 dimensions;
}
  
struct v2p
{
	float4 position : SV_POSITION;
	float2 texCoord : TEXCOORD0;
};

v2p VSMain(uint id : SV_VertexID, 
	float3 position : POSITION,
	float2 texCoord : TEXCOORD0)
{
	v2p result;

	result.position = float4(position, 1.0f);
	result.texCoord = texCoord;

	return result;
}

float4 PSMain(v2p input) : SV_TARGET
{
	uint2 coord = uint2(input.texCoord.x * dimensions.x, input.texCoord.y * dimensions.y);
	
	float4 tex = float4(0.0f, 0.0f, 0.0f, 0.0f);
  
#if SamplesMSAA == 1
	tex = msaaTexture.Load(coord, 0);
#else
	for (uint i = 0; i < SamplesMSAA; i++)
	{
		tex += msaaTexture.Load(coord, i);
	}
	tex *= 1.0f / SamplesMSAA;
#endif
		
	return tex;
}
