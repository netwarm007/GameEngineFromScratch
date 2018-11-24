#ifndef __STDCBUFFER_H__
#define __STDCBUFFER_H__

#define MAX_LIGHTS 100


#ifdef __cplusplus
	#include "portable.hpp"
	#include "geommath.hpp"
	#include "Guid.hpp"
	#include "SceneObjectLight.hpp"
	using namespace My;
	#define SEMANTIC(a) 
	#define REGISTER(x)
	#define unistruct struct
	#define SamplerState void

namespace My {
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
	#define Matrix2X2f row_major float2x2
	#define Matrix3X3f row_major float3x3
	#define Matrix4X4f row_major float4x4
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
};												// totle 256 bytes

unistruct PerFrameConstants REGISTER(b10)
{
	Matrix4X4f 	viewMatrix;						// 64 bytes
	Matrix4X4f 	projectionMatrix;				// 64 bytes
    Vector4f   	camPos;							// 16 bytes
	uint32_t  	numLights;						// 4 bytes
	float	    padding[3];						// 12 bytes
	Light 		lights[MAX_LIGHTS];   			// alignment = 64 bytes
};

unistruct PerBatchConstants REGISTER(b11)
{
	Matrix4X4f modelMatrix;						// 64 bytes
};

#ifdef __cplusplus
const size_t kSizePerFrameConstantBuffer = ALIGN(sizeof(PerFrameConstants) + sizeof(Light) * MAX_LIGHTS, 256); // CB size is required to be 256-byte aligned.
const size_t kSizePerBatchConstantBuffer = ALIGN(sizeof(PerBatchConstants), 256); // CB size is required to be 256-byte aligned.
#endif

struct a2v
{
    Vector3f inputPosition    SEMANTIC(POSITION);
    Vector2f inputUV          SEMANTIC(TEXCOORD);
    Vector3f inputNormal      SEMANTIC(NORMAL);
    Vector3f inputTangent     SEMANTIC(TANGENT);
    Vector3f inputBiTangent   SEMANTIC(BITANGENT);
};

#ifdef __cplusplus
struct material_textures
{
	int32_t diffuseMap = -1;
	int32_t normalMap = -1;
	int32_t metalicMap = -1;
	int32_t roughnessMap = -1;
	int32_t aoMap = -1;
	int32_t heightMap = -1;
};

struct global_textures
{
	int32_t brdfLUT;
};

struct frame_textures
{
	int32_t shadowMap;
	int32_t globalShadowMap;
	int32_t cubeShadowMap;
	int32_t skybox;
	int32_t terrainHeightMap;
};
#else
Texture2D diffuseMap 			REGISTER(t0);
Texture2D normalMap  			REGISTER(t1);
Texture2D metalicMap 			REGISTER(t2);
Texture2D roughnessMap 			REGISTER(t3);
Texture2D aoMap      			REGISTER(t4);
Texture2D heightMap				REGISTER(t5);
Texture2D brdfLUT    			REGISTER(t6);
Texture2DArray shadowMap  		REGISTER(t7);
Texture2DArray globalShadowMap 	REGISTER(t8);
TextureCubeArray cubeShadowMap 	REGISTER(t9);
TextureCubeArray skybox     	REGISTER(t10);
Texture2D terrainHeightMap		REGISTER(t11);

// samplers
SamplerState samp0 REGISTER(s0);
#endif

#ifdef __cplusplus
} // namespace My
#endif
#endif // !__STDCBUFFER_H__
