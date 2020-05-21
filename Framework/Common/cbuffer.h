#ifndef __STDCBUFFER_H__
#define __STDCBUFFER_H__

#define MAX_LIGHTS 100

#include "config.h"

#ifdef __cplusplus
#include "Guid.hpp"
#include "SceneObjectLight.hpp"
#include "geommath.hpp"
using namespace My;
#define SEMANTIC(a)
#define REGISTER(x)
#define unistruct struct
#define SamplerState void

namespace My {
enum LightType { Omni = 0, Spot = 1, Infinity = 2, Area = 3 };
#else
#define SEMANTIC(a) : a
#define REGISTER(x) : register(x)
#define unistruct cbuffer
#define int32_t int
#define Guid int4
#define Vector2f float2
#define Vector3f float3
#define Vector4f float4
#define Matrix2X2f row_major float2x2
#define Matrix3X3f row_major float3x3
#define Matrix4X4f row_major float4x4
#define LightType int
#define AttenCurveType int
#endif

struct Light {
    Matrix4X4f lightViewMatrix;               // 64 bytes
    Matrix4X4f lightProjectionMatrix;         // 64 bytes
    float lightIntensity;                     // 4 bytes
    LightType lightType;                      // 4 bytes
    int lightCastShadow;                      // 4 bytes
    int lightShadowMapIndex;                  // 4 bytes
    AttenCurveType lightAngleAttenCurveType;  // 4 bytes
    AttenCurveType lightDistAttenCurveType;   // 4 bytes
    Vector2f lightSize;                       // 8 bytes
    Guid lightGuid;                           // 16 bytes
    Vector4f lightPosition;                   // 16 bytes
    Vector4f lightColor;                      // 16 bytes
    Vector4f lightDirection;                  // 16 bytes
    Vector4f lightDistAttenCurveParams[2];    // 32 bytes
    Vector4f lightAngleAttenCurveParams[2];   // 32 bytes
};                                            // totle 288 bytes

unistruct PerFrameConstants REGISTER(b10) {
    Matrix4X4f viewMatrix;        // 64 bytes
    Matrix4X4f projectionMatrix;  // 64 bytes
    Vector4f camPos;              // 16 bytes
    int32_t numLights;            // 4 bytes
};                                // totle 148 bytes

unistruct PerBatchConstants REGISTER(b11) {
    Matrix4X4f modelMatrix;  // 64 bytes
};

unistruct LightInfo REGISTER(b12) {
    struct Light lights[MAX_LIGHTS];  // 288 bytes * MAX_LIGHTS
};

unistruct DebugConstants REGISTER(b13) {
    Vector4f front_color;  // 16 bytes
    Vector4f back_color;   // 16 bytes
    float layer_index;     // 4 bytes
    float mip_level;       // 4 bytes
    float line_width;      // 4 bytes
    float padding0;        // 4 bytes
};                         // 48 bytes

unistruct ShadowMapConstants REGISTER(b13) {
    int32_t light_index;          // 4 bytes
    float shadowmap_layer_index;  // 4 bytes
    float near_plane;             // 4 bytes
    float far_plane;              // 4 bytes
};                                // 16 bytes

#ifdef __cplusplus
const size_t kSizePerFrameConstantBuffer =
    ALIGN(sizeof(PerFrameConstants),
          256);  // CB size is required to be 256-byte aligned.
const size_t kSizePerBatchConstantBuffer =
    ALIGN(sizeof(PerBatchConstants),
          256);  // CB size is required to be 256-byte aligned.
const size_t kSizeLightInfo = ALIGN(
    sizeof(LightInfo), 256);  // CB size is required to be 256-byte aligned.
const size_t kSizeDebugConstantBuffer =
    ALIGN(sizeof(DebugConstants),
          256);  // CB size is required to be 256-byte aligned.
const size_t kSizeShadowMapConstantBuffer =
    ALIGN(sizeof(ShadowMapConstants),
          256);  // CB size is required to be 256-byte aligned.

enum A2V_TYPES {
    A2V_TYPES_NONE,
    A2V_TYPES_FULL,
    A2V_TYPES_SIMPLE,
    A2V_TYPES_POS_ONLY,
    A2V_TYPES_CUBE
};
#endif

struct a2v {
    Vector3f inputPosition SEMANTIC(POSITION);
    Vector3f inputNormal SEMANTIC(NORMAL);
    Vector2f inputUV SEMANTIC(TEXCOORD);
    Vector3f inputTangent SEMANTIC(TANGENT);
};

struct a2v_simple {
    Vector3f inputPosition SEMANTIC(POSITION);
    Vector2f inputUV SEMANTIC(TEXCOORD);
};

struct a2v_pos_only {
    Vector3f inputPosition SEMANTIC(POSITION);
};

struct a2v_cube {
    Vector3f inputPosition SEMANTIC(POSITION);
    Vector3f inputUVW SEMANTIC(TEXCOORD);
};

#ifdef __cplusplus
struct material_textures {
    int32_t diffuseMap = -1;
    int32_t normalMap = -1;
    int32_t metallicMap = -1;
    int32_t roughnessMap = -1;
    int32_t aoMap = -1;
    int32_t heightMap = -1;
};

struct global_textures {
    int32_t brdfLUT = -1;
    int32_t skybox = -1;
    int32_t terrainHeightMap = -1;
};

struct frame_textures {
    int32_t shadowMap = -1;
    int32_t shadowMapCount = 0;

    int32_t globalShadowMap = -1;
    int32_t globalShadowMapCount = 0;

    int32_t cubeShadowMap = -1;
    int32_t cubeShadowMapCount = 0;
};
#else
Texture2D diffuseMap REGISTER(t0);
Texture2D normalMap REGISTER(t1);
Texture2D metallicMap REGISTER(t2);
Texture2D roughnessMap REGISTER(t3);
Texture2D aoMap REGISTER(t4);
Texture2D heightMap REGISTER(t5);
Texture2D brdfLUT REGISTER(t6);
Texture2DArray shadowMap REGISTER(t7);
Texture2DArray globalShadowMap REGISTER(t8);
#if defined(OS_WEBASSEMBLY)
Texture2DArray cubeShadowMap REGISTER(t9);
Texture2DArray skybox REGISTER(t10);
#else
TextureCubeArray cubeShadowMap REGISTER(t9);
TextureCubeArray skybox REGISTER(t10);
#endif
Texture2D terrainHeightMap REGISTER(t11);

// samplers
SamplerState samp0 REGISTER(s0);
#endif

#ifdef __cplusplus
}  // namespace My
#endif
#endif  // !__STDCBUFFER_H__
