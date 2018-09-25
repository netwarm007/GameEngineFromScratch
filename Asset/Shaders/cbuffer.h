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
	float       lightIntensity;               	// 4 bytes
	LightType   lightType;                    	// 4 bytes
	int			lightCastShadow;				// 4 bytes
	int         lightShadowMapIndex;			// 4 bytes
	AttenCurveType lightAngleAttenCurveType;  	// 4 bytes
	AttenCurveType lightDistAttenCurveType; 	// 4 bytes
	Vector2f    lightSize;               		// 8 bytes
	Guid        lightGuid;                    	// 16 bytes
	Vector4f    lightPosition;   				// 16 bytes
	Vector4f    lightColor;   					// 16 bytes
	Vector4f    lightDirection;   				// 16 bytes
	Vector4f    lightDistAttenCurveParams[2]; 	// 32 bytes
	Vector4f    lightAngleAttenCurveParams[2];	// 32 bytes
	Matrix4X4f  lightVP;						// 64 bytes
	Vector4f    padding[2];						// 32 bytes
};												// totle 265 bytes

unistruct PerFrameConstants REGISTER(b0)
{
	Matrix4X4f 	viewMatrix;						// 64 bytes
	Matrix4X4f 	projectionMatrix;				// 64 bytes
    Vector4f   	camPos;							// 16 bytes
	uint32_t  	numLights;						// 4 bytes
	float	    padding[3];						// 12 bytes
	Light 		lights[MAX_LIGHTS];   			// alignment = 64 bytes
};

unistruct PerBatchConstants REGISTER(b1)
{
	Matrix4X4f modelMatrix;						// 64 bytes
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
