////////////////////////////////////////////////////////////////////////////////
// Filename: basic.ps 
////////////////////////////////////////////////////////////////////////////////
#version 150


/////////////////////
// CONSTANTS       //
/////////////////////
// per frame
uniform vec3   lightPosition;
uniform vec4   lightColor;
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
void main(void)
{
    // vertex normal
    vec3 N = normalize(normal.xyz);

    vec3 L = normalize((viewMatrix * worldMatrix * vec4(lightPosition, 1.0f)).xyz - v.xyz);
    vec3 R = normalize(2.0f * clamp(dot(L,N), 0.0f, 1.0f) * N - L);
    vec3 V = normalize(v.xyz);
    /*
    if (usingNormalMap)
    {
        N = perturb_normal(N, V, uv);
    }
    */
    if (usingDiffuseMap)
        outputColor = vec4(ambientColor.rgb + lightColor.rgb * (texture(diffuseMap, uv).rgb * clamp(dot(N, L), 0.0f, 1.0f) + specularColor.rgb * pow(clamp(dot(R, V), 0.0f, 1.0f), specularPower)), 1.0f); 
    else
        outputColor = vec4(ambientColor.rgb + lightColor.rgb * (diffuseColor.rgb * clamp(dot(N, L), 0.0f, 1.0f) + specularColor.rgb * pow(clamp(dot(R,V), 0.0f, 1.0f), specularPower)), 1.0f); 
}

