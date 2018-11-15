#version 400

struct vert_output
{
    vec4 position;
    vec4 normal;
    vec4 normal_world;
    vec4 v;
    vec4 v_world;
    vec2 uv;
    mat3 TBN;
    vec3 v_tangent;
    vec3 camPos_tangent;
};

struct Light
{
    float lightIntensity;
    uint lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    uint lightAngleAttenCurveType;
    uint lightDistAttenCurveType;
    vec2 lightSize;
    uvec4 lightGuid;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(std140) uniform PerFrameConstants
{
    layout(row_major) mat4 viewMatrix;
    layout(row_major) mat4 projectionMatrix;
    vec4 camPos;
    uint numLights;
    float padding[3];
    Light lights[100];
} _551;

uniform sampler2D SPIRV_Cross_CombineddiffuseMapsamp0;
uniform samplerCubeArray SPIRV_Cross_CombinedcubeShadowMapsamp0;
uniform sampler2DArray SPIRV_Cross_CombinedshadowMapsamp0;
uniform sampler2DArray SPIRV_Cross_CombinedglobalShadowMapsamp0;
uniform samplerCubeArray SPIRV_Cross_Combinedskyboxsamp0;

in vec4 input_normal;
in vec4 input_normal_world;
in vec4 input_v;
in vec4 input_v_world;
in vec2 input_uv;
in mat3 input_TBN;
in vec3 input_v_tangent;
in vec3 input_camPos_tangent;
layout(location = 0) out vec4 _entryPointOutput;

float _142;

vec3 projectOnPlane(vec3 _point, vec3 center_of_plane, vec3 normal_of_plane)
{
    return _point - (normal_of_plane * dot(_point - center_of_plane, normal_of_plane));
}

float linear_interpolate(float t, float begin, float end)
{
    if (t < begin)
    {
        return 1.0;
    }
    else
    {
        if (t > end)
        {
            return 0.0;
        }
        else
        {
            return (end - t) / (end - begin);
        }
    }
}

float apply_atten_curve(float dist, int atten_curve_type, vec4 atten_params[2])
{
    float atten = 1.0;
    switch (atten_curve_type)
    {
        case 1:
        {
            float begin_atten = atten_params[0].x;
            float end_atten = atten_params[0].y;
            float param = dist;
            float param_1 = begin_atten;
            float param_2 = end_atten;
            float param_3 = param;
            float param_4 = param_1;
            float param_5 = param_2;
            atten = linear_interpolate(param_3, param_4, param_5);
            break;
        }
        case 2:
        {
            float begin_atten_1 = atten_params[0].x;
            float end_atten_1 = atten_params[0].y;
            float param_3_1 = dist;
            float param_4_1 = begin_atten_1;
            float param_5_1 = end_atten_1;
            float param_6 = param_3_1;
            float param_7 = param_4_1;
            float param_8 = param_5_1;
            float tmp = linear_interpolate(param_6, param_7, param_8);
            atten = (3.0 * pow(tmp, 2.0)) - (2.0 * pow(tmp, 3.0));
            break;
        }
        case 3:
        {
            float scale = atten_params[0].x;
            float offset = atten_params[0].y;
            float kl = atten_params[0].z;
            float kc = atten_params[0].w;
            atten = clamp((scale / ((kl * dist) + (kc * scale))) + offset, 0.0, 1.0);
            break;
        }
        case 4:
        {
            float scale_1 = atten_params[0].x;
            float offset_1 = atten_params[0].y;
            float kq = atten_params[0].z;
            float kl_1 = atten_params[0].w;
            float kc_1 = atten_params[1].x;
            atten = clamp(pow(scale_1, 2.0) / ((((kq * pow(dist, 2.0)) + ((kl_1 * dist) * scale_1)) + (kc_1 * pow(scale_1, 2.0))) + offset_1), 0.0, 1.0);
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

bool isAbovePlane(vec3 _point, vec3 center_of_plane, vec3 normal_of_plane)
{
    return dot(_point - center_of_plane, normal_of_plane) > 0.0;
}

vec3 linePlaneIntersect(vec3 line_start, vec3 line_dir, vec3 center_of_plane, vec3 normal_of_plane)
{
    return line_start + (line_dir * (dot(center_of_plane - line_start, normal_of_plane) / dot(line_dir, normal_of_plane)));
}

vec3 apply_areaLight(Light light, vert_output _input)
{
    vec3 N = normalize(_input.normal.xyz);
    vec3 right = normalize((_551.viewMatrix * vec4(1.0, 0.0, 0.0, 0.0)).xyz);
    vec3 pnormal = normalize((_551.viewMatrix * light.lightDirection).xyz);
    vec3 ppos = (_551.viewMatrix * light.lightPosition).xyz;
    vec3 up = normalize(cross(pnormal, right));
    right = normalize(cross(up, pnormal));
    float width = light.lightSize.x;
    float height = light.lightSize.y;
    vec3 param = _input.v.xyz;
    vec3 param_1 = ppos;
    vec3 param_2 = pnormal;
    vec3 projection = projectOnPlane(param, param_1, param_2);
    vec3 dir = projection - ppos;
    vec2 diagonal = vec2(dot(dir, right), dot(dir, up));
    vec2 nearest2D = vec2(clamp(diagonal.x, -width, width), clamp(diagonal.y, -height, height));
    vec3 nearestPointInside = (ppos + (right * nearest2D.x)) + (up * nearest2D.y);
    vec3 L = nearestPointInside - _input.v.xyz;
    float lightToSurfDist = length(L);
    L = normalize(L);
    float param_3 = lightToSurfDist;
    int param_4 = int(light.lightDistAttenCurveType);
    vec4 param_5[2] = light.lightDistAttenCurveParams;
    float atten = apply_atten_curve(param_3, param_4, param_5);
    vec3 linearColor = vec3(0.0);
    float pnDotL = dot(pnormal, -L);
    float nDotL = dot(N, L);
    vec3 param_6 = _input.v.xyz;
    vec3 param_7 = ppos;
    vec3 param_8 = pnormal;
    if ((nDotL > 0.0) && isAbovePlane(param_6, param_7, param_8))
    {
        vec3 V = normalize(-_input.v.xyz);
        vec3 R = normalize((N * (2.0 * dot(V, N))) - V);
        vec3 R2 = normalize((N * (2.0 * dot(L, N))) - L);
        vec3 param_9 = _input.v.xyz;
        vec3 param_10 = R;
        vec3 param_11 = ppos;
        vec3 param_12 = pnormal;
        vec3 E = linePlaneIntersect(param_9, param_10, param_11, param_12);
        float specAngle = clamp(dot(-R, pnormal), 0.0, 1.0);
        vec3 dirSpec = E - ppos;
        vec2 dirSpec2D = vec2(dot(dirSpec, right), dot(dirSpec, up));
        vec2 nearestSpec2D = vec2(clamp(dirSpec2D.x, -width, width), clamp(dirSpec2D.y, -height, height));
        float specFactor = 1.0 - clamp(length(nearestSpec2D - dirSpec2D), 0.0, 1.0);
        vec3 admit_light = light.lightColor.xyz * (light.lightIntensity * atten);
        linearColor = (texture(SPIRV_Cross_CombineddiffuseMapsamp0, _input.uv).xyz * nDotL) * pnDotL;
        linearColor += (((vec3(0.800000011920928955078125) * pow(clamp(dot(R2, V), 0.0, 1.0), 50.0)) * specFactor) * specAngle);
        linearColor *= admit_light;
    }
    return linearColor;
}

float shadow_test(vec4 p, Light light, float cosTheta)
{
    vec4 v_light_space = light.lightVP * p;
    v_light_space /= vec4(v_light_space.w);
    float visibility = 1.0;
    if (light.lightShadowMapIndex != (-1))
    {
        float bias = 0.0005000000237487256526947021484375 * tan(acos(cosTheta));
        bias = clamp(bias, 0.0, 0.00999999977648258209228515625);
        float near_occ;
        switch (light.lightType)
        {
            case 0:
            {
                vec3 L = p.xyz - light.lightPosition.xyz;
                near_occ = texture(SPIRV_Cross_CombinedcubeShadowMapsamp0, vec4(L, float(light.lightShadowMapIndex))).x;
                if ((length(L) - (near_occ * 10.0)) > bias)
                {
                    visibility -= 0.87999999523162841796875;
                }
                break;
            }
            case 1:
            {
                v_light_space *= mat4(vec4(0.5, 0.0, 0.0, 0.0), vec4(0.0, 0.5, 0.0, 0.0), vec4(0.0, 0.0, 0.5, 0.0), vec4(0.5, 0.5, 0.5, 1.0));
                for (int i = 0; i < 4; i++)
                {
                    mat2x4 indexable = mat2x4(vec4(-0.94201624393463134765625, -0.39906215667724609375, 0.94558608531951904296875, -0.768907248973846435546875), vec4(-0.094184100627899169921875, -0.929388701915740966796875, 0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = texture(SPIRV_Cross_CombinedshadowMapsamp0, vec3(v_light_space.xy + (vec2(indexable[i].xy) / vec2(700.0)), float(light.lightShadowMapIndex))).x;
                    if ((v_light_space.z - near_occ) > bias)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
            case 2:
            {
                v_light_space *= mat4(vec4(0.5, 0.0, 0.0, 0.0), vec4(0.0, 0.5, 0.0, 0.0), vec4(0.0, 0.0, 0.5, 0.0), vec4(0.5, 0.5, 0.5, 1.0));
                for (int i_1 = 0; i_1 < 4; i_1++)
                {
                    mat2x4 indexable_1 = mat2x4(vec4(-0.94201624393463134765625, -0.39906215667724609375, 0.94558608531951904296875, -0.768907248973846435546875), vec4(-0.094184100627899169921875, -0.929388701915740966796875, 0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = texture(SPIRV_Cross_CombinedglobalShadowMapsamp0, vec3(v_light_space.xy + (vec2(indexable_1[i_1].xy) / vec2(700.0)), float(light.lightShadowMapIndex))).x;
                    if ((v_light_space.z - near_occ) > bias)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
            case 3:
            {
                v_light_space *= mat4(vec4(0.5, 0.0, 0.0, 0.0), vec4(0.0, 0.5, 0.0, 0.0), vec4(0.0, 0.0, 0.5, 0.0), vec4(0.5, 0.5, 0.5, 1.0));
                for (int i_2 = 0; i_2 < 4; i_2++)
                {
                    mat2x4 indexable_2 = mat2x4(vec4(-0.94201624393463134765625, -0.39906215667724609375, 0.94558608531951904296875, -0.768907248973846435546875), vec4(-0.094184100627899169921875, -0.929388701915740966796875, 0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = texture(SPIRV_Cross_CombinedshadowMapsamp0, vec3(v_light_space.xy + (vec2(indexable_2[i_2].xy) / vec2(700.0)), float(light.lightShadowMapIndex))).x;
                    if ((v_light_space.z - near_occ) > bias)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
        }
    }
    return visibility;
}

vec3 apply_light(Light light, vert_output _input)
{
    vec3 N = normalize(_input.normal.xyz);
    vec3 light_dir = normalize((_551.viewMatrix * light.lightDirection).xyz);
    vec3 L;
    if (light.lightPosition.w == 0.0)
    {
        L = -light_dir;
    }
    else
    {
        L = (_551.viewMatrix * light.lightPosition).xyz - _input.v.xyz;
    }
    float lightToSurfDist = length(L);
    L = normalize(L);
    float cosTheta = clamp(dot(N, L), 0.0, 1.0);
    vec4 param = _input.v_world;
    Light param_1 = light;
    float param_2 = cosTheta;
    float visibility = shadow_test(param, param_1, param_2);
    float lightToSurfAngle = acos(dot(L, -light_dir));
    float param_3 = lightToSurfAngle;
    int param_4 = int(light.lightAngleAttenCurveType);
    vec4 param_5[2] = light.lightAngleAttenCurveParams;
    float atten = apply_atten_curve(param_3, param_4, param_5);
    float param_6 = lightToSurfDist;
    int param_7 = int(light.lightDistAttenCurveType);
    vec4 param_8[2] = light.lightDistAttenCurveParams;
    atten *= apply_atten_curve(param_6, param_7, param_8);
    vec3 R = normalize((N * (2.0 * dot(L, N))) - L);
    vec3 V = normalize(-_input.v.xyz);
    vec3 admit_light = light.lightColor.xyz * (light.lightIntensity * atten);
    vec3 linearColor = texture(SPIRV_Cross_CombineddiffuseMapsamp0, _input.uv).xyz * cosTheta;
    if (visibility > 0.20000000298023223876953125)
    {
        linearColor += (vec3(0.800000011920928955078125) * pow(clamp(dot(R, V), 0.0, 1.0), 50.0));
    }
    linearColor *= admit_light;
    return linearColor * visibility;
}

vec3 exposure_tone_mapping(vec3 color)
{
    return vec3(1.0) - exp((-color) * 1.0);
}

vec3 gamma_correction(vec3 color)
{
    return pow(color, vec3(0.4545454680919647216796875));
}

vec4 _basic_frag_main(vert_output _input)
{
    vec3 linearColor = vec3(0.0);
    for (int i = 0; uint(i) < _551.numLights; i++)
    {
        if (_551.lights[i].lightType == 3u)
        {
            Light arg;
            arg.lightIntensity = _551.lights[i].lightIntensity;
            arg.lightType = _551.lights[i].lightType;
            arg.lightCastShadow = _551.lights[i].lightCastShadow;
            arg.lightShadowMapIndex = _551.lights[i].lightShadowMapIndex;
            arg.lightAngleAttenCurveType = _551.lights[i].lightAngleAttenCurveType;
            arg.lightDistAttenCurveType = _551.lights[i].lightDistAttenCurveType;
            arg.lightSize = _551.lights[i].lightSize;
            arg.lightGuid = _551.lights[i].lightGuid;
            arg.lightPosition = _551.lights[i].lightPosition;
            arg.lightColor = _551.lights[i].lightColor;
            arg.lightDirection = _551.lights[i].lightDirection;
            arg.lightDistAttenCurveParams[0] = _551.lights[i].lightDistAttenCurveParams[0];
            arg.lightDistAttenCurveParams[1] = _551.lights[i].lightDistAttenCurveParams[1];
            arg.lightAngleAttenCurveParams[0] = _551.lights[i].lightAngleAttenCurveParams[0];
            arg.lightAngleAttenCurveParams[1] = _551.lights[i].lightAngleAttenCurveParams[1];
            arg.lightVP = _551.lights[i].lightVP;
            arg.padding[0] = _551.lights[i].padding[0];
            arg.padding[1] = _551.lights[i].padding[1];
            vert_output param = _input;
            linearColor += apply_areaLight(arg, param);
        }
        else
        {
            Light arg_1;
            arg_1.lightIntensity = _551.lights[i].lightIntensity;
            arg_1.lightType = _551.lights[i].lightType;
            arg_1.lightCastShadow = _551.lights[i].lightCastShadow;
            arg_1.lightShadowMapIndex = _551.lights[i].lightShadowMapIndex;
            arg_1.lightAngleAttenCurveType = _551.lights[i].lightAngleAttenCurveType;
            arg_1.lightDistAttenCurveType = _551.lights[i].lightDistAttenCurveType;
            arg_1.lightSize = _551.lights[i].lightSize;
            arg_1.lightGuid = _551.lights[i].lightGuid;
            arg_1.lightPosition = _551.lights[i].lightPosition;
            arg_1.lightColor = _551.lights[i].lightColor;
            arg_1.lightDirection = _551.lights[i].lightDirection;
            arg_1.lightDistAttenCurveParams[0] = _551.lights[i].lightDistAttenCurveParams[0];
            arg_1.lightDistAttenCurveParams[1] = _551.lights[i].lightDistAttenCurveParams[1];
            arg_1.lightAngleAttenCurveParams[0] = _551.lights[i].lightAngleAttenCurveParams[0];
            arg_1.lightAngleAttenCurveParams[1] = _551.lights[i].lightAngleAttenCurveParams[1];
            arg_1.lightVP = _551.lights[i].lightVP;
            arg_1.padding[0] = _551.lights[i].padding[0];
            arg_1.padding[1] = _551.lights[i].padding[1];
            vert_output param_1 = _input;
            linearColor += apply_light(arg_1, param_1);
        }
    }
    linearColor += (texture(SPIRV_Cross_Combinedskyboxsamp0, vec4(_input.normal_world.xyz, 0.0)).xyz * vec3(0.20000000298023223876953125));
    vec3 param_2 = linearColor;
    linearColor = exposure_tone_mapping(param_2);
    vec3 param_3 = linearColor;
    return vec4(gamma_correction(param_3), 1.0);
}

void main()
{
    vert_output _input;
    _input.position = gl_FragCoord;
    _input.normal = input_normal;
    _input.normal_world = input_normal_world;
    _input.v = input_v;
    _input.v_world = input_v_world;
    _input.uv = input_uv;
    _input.TBN = input_TBN;
    _input.v_tangent = input_v_tangent;
    _input.camPos_tangent = input_camPos_tangent;
    vert_output param = _input;
    _entryPointOutput = _basic_frag_main(param);
}

