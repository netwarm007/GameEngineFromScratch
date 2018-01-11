#ifndef __STDCBUFFER_H__
#define __STDCBUFFER_H__

struct a2v
{
	float3 Position		: POSITION;
	float3 Normal		: NORMAL;
	float4 Tangent		: TANGENT;
	float2 TextureUV	: TEXCOORD;
};

cbuffer Constants : register(b0)
{
	float4x4 m_modelView;
	float4x4 m_modelViewProjection;
	float4   m_lightPosition;
	float4   m_lightColor;
	float4   m_ambientColor;
	float4   m_lightAttenuation;
};

#endif // !__STDCBUFFER_H__
