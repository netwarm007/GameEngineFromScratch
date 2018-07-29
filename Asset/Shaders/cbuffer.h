#ifndef __STDCBUFFER_H__
#define __STDCBUFFER_H__

#define MAX_LIGHTS 55

struct a2v
{
	float3 Position		: POSITION;
	float3 Normal		: NORMAL;
	float2 TextureUV	: TEXCOORD;
};

struct Light{
	float4		m_lightPosition;
	float4		m_lightColor;
	float4		m_lightDirection;
	float2	    m_lightSize;
	float       m_lightIntensity;
	uint		m_lightDistAttenCurveType;
	float       m_lightDistAttenCurveParams_0;
	float       m_lightDistAttenCurveParams_1;
	float       m_lightDistAttenCurveParams_2;
	float       m_lightDistAttenCurveParams_3;
	float       m_lightDistAttenCurveParams_4;
	uint		m_lightAngleAttenCurveType;
	float       m_lightAngleAttenCurveParams_0;
	float       m_lightAngleAttenCurveParams_1;
	float       m_lightAngleAttenCurveParams_2;
	float       m_lightAngleAttenCurveParams_3;
	float       m_lightAngleAttenCurveParams_4;
};

cbuffer PerFrameConstants : register(b0)
{
    float4x4 m_worldMatrix;
	float4x4 m_viewMatrix;
	float4x4 m_projectionMatrix;
    float3 ambientColor;
	Light m_lights[MAX_LIGHTS];
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

uint numLights : register(b2);

#endif // !__STDCBUFFER_H__
