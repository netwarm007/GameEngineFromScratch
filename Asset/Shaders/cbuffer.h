#ifndef __STDCBUFFER_H__
#define __STDCBUFFER_H__

#define MAX_LIGHTS 100

#ifdef __cplusplus
	#include "geommath.hpp"
	#include "Guid.hpp"
	#include "SceneObjectLight.hpp"
	using namespace My;
	#define SEMANTIC(a) 
	#define REGISTER(x)
	#define unistruct struct

	enum LightType {
		Omni     = 0,
		Spot     = 1,
		Infinity = 2,
		Area     = 3
	};
#else
	#define SEMANTIC(a) : a 
	#define REGISTER(x) : register(x)
	#define unistruct cbuffer
	#define uint32_t uint
	#define Guid uint4 
	#define Vector2f float2
	#define Vector3f float3
	#define Vector4f float4
	#define Matrix2X2f float2x2
	#define Matrix3X3f float3x3
	#define Matrix4X4f float4x4
	#define LightType uint
	#define AttenCurveType uint
#endif

struct Light{
	Guid        m_lightGuid;                    // 16 bytes
	LightType   m_lightType;                    // 4 bytes
	Vector4f    m_lightPosition;   				// 16 bytes
	Vector4f    m_lightColor;   				// 16 bytes
	Vector4f    m_lightDirection;   			// 16 bytes
	Vector2f    m_lightSize;               		// 8 bytes
	float       m_lightIntensity;               // 4 bytes
	AttenCurveType m_lightDistAttenCurveType; 	// 4 bytes
	float       m_lightDistAttenCurveParams[8]; // 32 bytes
	AttenCurveType m_lightAngleAttenCurveType;  // 4 bytes
	float       m_lightAngleAttenCurveParams[8];// 32 bytes
	int 		m_lightCastShadow;				// 4 bytes
	int         m_lightShadowMapIndex;			// 4 bytes
	Matrix4X4f  m_lightVP;						// 64 byptes
};

unistruct PerFrameConstants REGISTER(b0)
{
	Matrix4X4f 	m_viewMatrix;
	Matrix4X4f 	m_projectionMatrix;
    Vector4f   	ambientColor;
    Vector4f   	camPos;
	uint32_t  	numLights;
	Light m_lights[MAX_LIGHTS];
};

unistruct PerBatchConstants REGISTER(b1)
{
	Matrix4X4f modelMatrix;
    Vector4f diffuseColor;
    Vector4f specularColor;
	float specularPower;
	float metallic;
	float roughness;
	float ao;

	bool usingDiffuseMap;
	bool usingNormalMap;
	bool usingMetallicMap;
	bool usingRoughnessMap;
	bool usingAoMap;
};

#ifndef __cplusplus
// samplers
SamplerState samp0 : register(s0);

// textures
Texture2D diffuseMap : register(t0);
Texture2D normalMap  : register(t1);
Texture2D metalicMap : register(t2);
Texture2D roughnessMap : register(t3);
Texture2D aoMap      : register(t4);
Texture2D brdfLUT    : register(t5);

Texture2DArray shadowMap  		: register(t6);
Texture2DArray globalShadowMap 	: register(t7);
TextureCubeArray cubeShadowMap 	: register(t8);
TextureCubeArray skybox     	: register(t9);
#endif

#endif // !__STDCBUFFER_H__
