#ifndef __STDCBUFFER_H__
#define __STDCBUFFER_H__

struct a2v
{
	float3 Position		: POSITION;
	float3 Normal		: NORMAL;
	float2 TextureUV	: TEXCOORD;
};

cbuffer PerFrameConstants : register(b0)
{
    float4x4 m_worldMatrix;
	float4x4 m_viewMatrix;
	float4x4 m_projectionMatrix;
	float3   m_lightPosition;
	float4   m_lightColor;
    float3 ambientColor;
};

cbuffer PerBatchConstants : register(b1)
{
	float4x4 objectMatrix;
    float4 diffuseColor;
    float4 specularColor;
    float specularPower;
	bool usingDiffuseMap;
	bool usingNormalMap;
};

#endif // !__STDCBUFFER_H__
