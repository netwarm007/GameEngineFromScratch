////////////////////////////////////////////////////////////////////////////////
// Filename: basic.ps 
////////////////////////////////////////////////////////////////////////////////
#version 150


/////////////////////
// CONSTANTS       //
/////////////////////
// per frame
uniform vec4 lightPosition;
uniform vec4 lightColor;
uniform vec3 lightDirection;
uniform float lightIntensity;
uniform int  lightDistAttenCurveType;
uniform float lightDistAttenCurveParams[5];
uniform int  lightAngleAttenCurveType;
uniform float lightAngleAttenCurveParams[5];

uniform mat4 worldMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

// per drawcall
uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float specularPower;

uniform bool usingDiffuseMap;
uniform bool usingNormalMap;

uniform sampler2D diffuseMap;
uniform sampler2D normalMap;

/////////////////////
// INPUT VARIABLES //
/////////////////////
in vec4 normal;
in vec4 v; 
in vec2 uv;

//////////////////////
// OUTPUT VARIABLES //
//////////////////////
out vec4 outputColor;

////////////////////////////////////////////////////////////////////////////////
// Pixel Shader
////////////////////////////////////////////////////////////////////////////////

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

float apply_atten_curve(float dist, int atten_type, float atten_params[5])
{
    float atten = 1.0f;

    switch(atten_type)
    {
        case 0: // linear
        {
            float begin_atten = atten_params[0];
            float end_atten = atten_params[1];
            atten = linear_interpolate(dist, begin_atten, end_atten);
            break;
        }
        case 1: // smooth
        {
            float begin_atten = atten_params[0];
            float end_atten = atten_params[1];
            float tmp = linear_interpolate(dist, begin_atten, end_atten);
            atten = 3.0f * pow(tmp, 2.0f) + 2.0f * pow(tmp, 3.0f);
            break;
        }
        case 2: // inverse
        {
            float scale = atten_params[0];
            float offset = atten_params[1];
            float kl = atten_params[2];
            float kc = atten_params[3];
            atten = clamp(scale / 
                (kl * dist + kc * scale) + offset, 
                0.0f, 1.0f);
            break;
        }
        case 3: // inverse square
        {
            float scale = atten_params[0];
            float offset = atten_params[1];
            float kq = atten_params[2];
            float kl = atten_params[3];
            float kc = atten_params[4];
            atten = clamp(pow(scale, 2.0f) / 
                (kq * pow(dist, 2.0f) + kl * dist * scale + kc * pow(scale, 2.0f) + offset), 
                0.0f, 1.0f);
            break;
        }
    }

    return atten;
}

void main(void)
{
    // vertex normal
    vec3 N = normalize(normal.xyz);

    vec3 L = (viewMatrix * worldMatrix * lightPosition).xyz - v.xyz;
    float lightToSurfDist = length(L);
    L = normalize(L);
    vec3 light_dir = normalize((viewMatrix * worldMatrix * vec4(lightDirection, 0.0f)).xyz);
    float lightToSurfAngle = acos(dot(L, light_dir));

    // angle attenuation
    float atten = apply_atten_curve(lightToSurfAngle, lightAngleAttenCurveType, lightAngleAttenCurveParams);

    // distance attenuation
    atten *= apply_atten_curve(lightToSurfDist, lightDistAttenCurveType, lightDistAttenCurveParams);

    vec3 R = normalize(2.0f * clamp(dot(L,N), 0.0f, 1.0f) * N - L);
    vec3 V = normalize(v.xyz);

    if (usingDiffuseMap)
        outputColor = vec4(ambientColor.rgb + lightIntensity * atten * lightColor.rgb * (texture(diffuseMap, uv).rgb * clamp(dot(N, L), 0.0f, 1.0f) + specularColor.rgb * pow(clamp(dot(R, V), 0.0f, 1.0f), specularPower)), 1.0f); 
    else
        outputColor = vec4(ambientColor.rgb + lightIntensity * atten * lightColor.rgb * (diffuseColor.rgb * clamp(dot(N, L), 0.0f, 1.0f) + specularColor.rgb * pow(clamp(dot(R,V), 0.0f, 1.0f), specularPower)), 1.0f); 
}

