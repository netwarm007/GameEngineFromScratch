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

// update per frame
uniform mat4 worldMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

// per drawcall
uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float specularPower;

uniform sampler2D defaultSampler;

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
    vec3 N = normalize(normal.xyz);
    vec3 L = normalize((viewMatrix * vec4(lightPosition, 1.0f)).xyz - v.xyz);
    vec3 R = normalize(2 * dot(L,N) * N - L);
    vec3 V = normalize(v.xyz);
    if (diffuseColor.r < 0)
        outputColor = vec4(ambientColor.rgb + lightColor.rgb * texture(defaultSampler, uv).rgb * dot(N, L) + specularColor.rgb * pow(clamp(dot(R, V), 0.0f, 1.0f), specularPower), 1.0f); 
    else
        outputColor = vec4(ambientColor.rgb + lightColor.rgb * diffuseColor.rgb * dot(N, L) + specularColor.rgb * pow(clamp(dot(R,V), 0.0f, 1.0f), specularPower), 1.0f); 
}

