#include "cbuffer.h"
#include "const.h.hlsl"

float3 projectOnPlane(float3 _point, float3 center_of_plane, float3 normal_of_plane)
{
    return _point - (normal_of_plane * dot(_point - center_of_plane, normal_of_plane));
}

bool isAbovePlane(float3 _point, float3 center_of_plane, float3 normal_of_plane)
{
    return dot(_point - center_of_plane, normal_of_plane) > 0.0f;
}

float3 linePlaneIntersect(float3 line_start, float3 line_dir, float3 center_of_plane, float3 normal_of_plane)
{
    return line_start + (line_dir * (dot(center_of_plane - line_start, normal_of_plane) / dot(line_dir, normal_of_plane)));
}

float linear_interpolate(float t, float begin, float end)
{
    if (t < begin)
    {
        return 1.0f;
    }
    else if (t > end)
    {
        return 0.0f;
    }
    else
    {
        return (end - t) / (end - begin);
    }
}

float apply_atten_curve(float dist, int atten_curve_type, float4 atten_params[2])
{
    float atten = 1.0f;
    switch (atten_curve_type)
    {
        case 1:
        {
            float begin_atten = atten_params[0].x;
            float end_atten = atten_params[0].y;
            float param = dist;
            float param_1 = begin_atten;
            float param_2 = end_atten;
            atten = linear_interpolate(param, param_1, param_2);
            break;
        }
        case 2:
        {
            float begin_atten_1 = atten_params[0].x;
            float end_atten_1 = atten_params[0].y;
            float param_3 = dist;
            float param_4 = begin_atten_1;
            float param_5 = end_atten_1;
            float tmp = linear_interpolate(param_3, param_4, param_5);
            atten = (3.0f * pow(tmp, 2.0f)) - (2.0f * pow(tmp, 3.0f));
            break;
        }
        case 3:
        {
            float scale = atten_params[0].x;
            float offset = atten_params[0].y;
            float kl = atten_params[0].z;
            float kc = atten_params[0].w;
            atten = clamp((scale / ((kl * dist) + (kc * scale))) + offset, 0.0f, 1.0f);
            break;
        }
        case 4:
        {
            float scale_1 = atten_params[0].x;
            float offset_1 = atten_params[0].y;
            float kq = atten_params[0].z;
            float kl_1 = atten_params[0].w;
            float kc_1 = atten_params[1].x;
            atten = clamp(pow(scale_1, 2.0f) / ((((kq * pow(dist, 2.0f)) + ((kl_1 * dist) * scale_1)) + (kc_1 * pow(scale_1, 2.0f))) + offset_1), 0.0f, 1.0f);
            break;
        }
        case 0:
        {
            break;
        }
        default:
        {
            break;
        }
    }
    return atten;
}

float shadow_test(float4 p, Light light, float cosTheta, int clip_space_type) {
    float4 v_light_space = mul(mul(p, light.lightViewMatrix), light.lightProjectionMatrix);
    v_light_space /= v_light_space.w.xxxx;

    float4x4 depth_bias; 
    if (clip_space_type == 0) {
        // for OpenGL clip space
        depth_bias = float4x4 (
            float4(0.5f, 0.0f, 0.0f, 0.0f),
            float4(0.0f, 0.5f, 0.0f, 0.0f),
            float4(0.0f, 0.0f, 0.5f, 0.0f),
            float4(0.5f, 0.5f, 0.5f, 1.0f)
        );
    } else {
        // for others
        depth_bias = float4x4 (
            float4(0.5f, 0.0f, 0.0f, 0.0f),
            float4(0.0f, 0.5f, 0.0f, 0.0f),
            float4(0.0f, 0.0f, 1.0f, 0.0f),
            float4(0.5f, 0.5f, 0.0f, 1.0f)
        );
    }

    const float4x2 poissonDisk = float4x2 (
        float2( -0.94201624f, -0.39906216f ),
        float2( 0.94558609f, -0.76890725f ),
        float2( -0.094184101f, -0.92938870f ),
        float2( 0.34495938f, 0.29387760f )
    );

    // shadow test
    float visibility = 1.0f;
    if (light.lightCastShadow) // the light cast shadow
    {
        float bias = (5e-4) * tan(acos(cosTheta)); // cosTheta is dot( n,l ), clamped between 0 and 1
        bias = clamp(bias, 0.0f, 0.01f);
        float near_occ;
        int i;
        switch (light.lightType)
        {
            case 0: // point
                // recalculate the v_light_space because we do not need to taking account of rotation
                {
                    float3 L = p.xyz - light.lightPosition.xyz;
                    near_occ = cubeShadowMap.Sample(samp0, float4(L, float(light.lightShadowMapIndex))).x;

                    if (length(L) - near_occ * 10.0f > bias)
                    {
                        // we are in the shadow
                        visibility -= 0.88f;
                    }
                }
                break;
            case 1: // spot
                // adjust from [-1, 1] to [0, 1]
                v_light_space = mul(v_light_space, depth_bias);
                for (i = 0; i < 4; i++)
                {
                    near_occ = shadowMap.Sample(samp0, float3(v_light_space.xy + (poissonDisk[i] / 700.0f.xx), float(light.lightShadowMapIndex))).x;

                    if (v_light_space.z - near_occ > bias)
                    {
                        // we are in the shadow
                        visibility -= 0.22f;
                    }
                }
                break;
            case 2: // infinity
                // adjust from [-1, 1] to [0, 1]
                v_light_space = mul(v_light_space, depth_bias);
                for (i = 0; i < 4; i++)
                {
                    near_occ = globalShadowMap.Sample(samp0, float3(v_light_space.xy + (poissonDisk[i] / 700.0f.xx), float(light.lightShadowMapIndex))).x;

                    if (v_light_space.z - near_occ > bias)
                    {
                        // we are in the shadow
                        visibility -= 0.22f;
                    }
                }
                break;
            case 3: // area
                // adjust from [-1, 1] to [0, 1]
                v_light_space = mul(v_light_space, depth_bias);
                for (i = 0; i < 4; i++)
                {
                    near_occ = shadowMap.Sample(samp0, float3(v_light_space.xy + (poissonDisk[i] / 700.0f.xx), float(light.lightShadowMapIndex))).x;

                    if (v_light_space.z - near_occ > bias)
                    {
                        // we are in the shadow
                        visibility -= 0.22f;
                    }
                }
                break;
        }
    }

    return visibility;
}

float3 reinhard_tone_mapping(float3 color)
{
    return color / (color + 1.0f.xxx);
}

float3 exposure_tone_mapping(float3 color)
{
    const float exposure = 1.0f;
    return 1.0f.xxx - exp(-color * exposure);
}

float3 gamma_correction(float3 color)
{
    const float gamma = 2.2f;
    return pow(max(color, 0.0f.xxx), (1.0f / gamma).xxx);
}

float3 inverse_gamma_correction(float3 color)
{
    const float gamma = 2.2f;
    return pow(max(color, 0.0f.xxx), gamma.xxx);
}

float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + ((1.0f.xxx - F0) * pow(1.0f - cosTheta, 5.0f));
}

float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    return F0 + ((max((1.0f - roughness).xxx, F0) - F0) * pow(1.0f - cosTheta, 5.0f));
}

float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0f);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;
	
    return num / denom;
}

float GeometrySchlickGGXDirect(float NdotV, float roughness)
{
    float r = (roughness + 1.0f);
    float k = (r*r) / 8.0f;

    float num   = NdotV;
    float denom = NdotV * (1.0f - k) + k;
	
    return num / denom;
}

float GeometrySmithDirect(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx2  = GeometrySchlickGGXDirect(NdotV, roughness);
    float ggx1  = GeometrySchlickGGXDirect(NdotL, roughness);
	
    return ggx1 * ggx2;
}

float GeometrySchlickGGXIndirect(float NdotV, float roughness)
{
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmithIndirect(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGXIndirect(NdotV, roughness);
    float ggx1 = GeometrySchlickGGXIndirect(NdotL, roughness);

    return ggx1 * ggx2;
}

#if !defined(OS_WEBASSEMBLY)
float RadicalInverse_VdC(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

float2 Hammersley(uint i, uint N)
{
    return float2(float(i)/float(N), RadicalInverse_VdC(i));
}  
#endif

float3 ImportanceSampleGGX(float2 Xi, float3 N, float roughness)
{
    float a = roughness*roughness;
	
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    // from spherical coordinates to cartesian coordinates
    float3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
	
    // from tangent-space vector to world-space sample vector
    float3 up        = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent   = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);
	
    float3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

float2 ParallaxMapping(float2 texCoords, float3 viewDir)
{ 
    const float height_scale = 0.10f;

    // number of depth layers
    const float minLayers = 8.0;
    const float maxLayers = 32.0;
    float numLayers = linear_interpolate(abs(dot(float3(0.0, 0.0, 1.0), viewDir)), maxLayers, minLayers);
    // calculate the size of each layer
    const float layerDepth = 1.0f / numLayers;
    // depth of current layer
    float currentLayerDepth = 0.0f;

    // get initial values
    float2  currentTexCoords     = texCoords;
    float currentDepthMapValue = 1.0f - heightMap.Sample(samp0, currentTexCoords).x;

    // the amount to shift the texture coordinates per layer (from vector P)
    const float2 P = viewDir.xy * height_scale; 
    const float2 deltaTexCoords = P / numLayers;
    
    while(currentLayerDepth < currentDepthMapValue)
    {
        // shift texture coordinates along direction of P
        currentTexCoords -= deltaTexCoords;
        // get depthmap value at current texture coordinates
        currentDepthMapValue = 1.0f - heightMap.Sample(samp0, currentTexCoords).x;  
        // get depth of next layer
        currentLayerDepth += layerDepth;  
    }

    // get texture coordinates before collision (reverse operations)
    float2 prevTexCoords = currentTexCoords + deltaTexCoords;

    // get depth after and before collision for linear interpolation
    float afterDepth  = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = 1.0f - heightMap.Sample(samp0, prevTexCoords).x - currentLayerDepth + layerDepth;
    
    // interpolation of texture coordinates
    float weight = afterDepth / (afterDepth - beforeDepth);
    float2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0f - weight);

    return finalTexCoords;  
} 

float4 project(float4 vertex){
    float4 result = mul(mul(vertex, viewMatrix), projectionMatrix);
    result /= result.w.xxxx;
    return result;
}

float2 screen_space(float4 vertex){
    return (clamp(vertex.xy, -1.3f, 1.3f) + 1.0f) * (float2(960.0f, 540.0f) * 0.5f);
}

bool offscreen(float4 vertex){
    if(vertex.z < -0.5f){
        return true;
    }   
    return any(
        (vertex.xy < -1.7f.xx) ||
        any(vertex.xy > 1.7f.xx)
    );  
}

float level(float2 v0, float2 v1){
     return clamp(distance(v0, v1)/2.0f, 1.0f, 64.0f);
}

float3 convert_xyz_to_cube_uv(float3 d)
{
    float3 d_abs = abs(d);
  
    bool3 isPositive;
    isPositive.x = d.x > 0 ? 1 : 0;
    isPositive.y = d.y > 0 ? 1 : 0;
    isPositive.z = d.z > 0 ? 1 : 0;
  
    float maxAxis = 0.0f, uc = 0.0f, vc = 0.0f;
    int index = 0;
  
    // POSITIVE X
    if (isPositive.x && d_abs.x >= d_abs.y && d_abs.x >= d_abs.z) {
        // u (0 to 1) goes from +y to -y
        // v (0 to 1) goes from -z to +z
        maxAxis = d_abs.x;
        uc = -d.z;
        vc = d.y;
        index = 0;
    }
    // NEGATIVE X
    if (!isPositive.x && d_abs.x >= d_abs.y && d_abs.x >= d_abs.z) {
        // u (0 to 1) goes from -y to +y
        // v (0 to 1) goes from -z to +z
        maxAxis = d_abs.x;
        uc = d.z;
        vc = d.y;
        index = 1;
    }
    // POSITIVE Y
    if (isPositive.y && d_abs.y >= d_abs.x && d_abs.y >= d_abs.z) {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from +z to -z
        maxAxis = d_abs.y;
        uc = d.x;
        vc = -d.z;
        index = 3;
    }
    // NEGATIVE Y
    if (!isPositive.y && d_abs.y >= d_abs.x && d_abs.y >= d_abs.z) {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from -z to +z
        maxAxis = d_abs.y;
        uc = d.x;
        vc = d.z;
        index = 2;
    }
    // POSITIVE Z
    if (isPositive.z && d_abs.z >= d_abs.x && d_abs.z >= d_abs.y) {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from +y to -y
        maxAxis = d_abs.z;
        uc = d.x;
        vc = d.y;
        index = 4;
    }
    // NEGATIVE Z
    if (!isPositive.z && d_abs.z >= d_abs.x && d_abs.z >= d_abs.y) {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from -y to +y
        maxAxis = d_abs.z;
        uc = -d.x;
        vc = d.y;
        index = 5;
    }

    // Convert range from -1 to 1 to 0 to 1
    float3 o;
    o.x = 0.5f * (uc / maxAxis + 1.0f);
    o.y = 0.5f * (vc / maxAxis + 1.0f);
    o.z = (float)index;

    return o;
}
