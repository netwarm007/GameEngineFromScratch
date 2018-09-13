in vec3 UVW;

out vec3 color;

uniform samplerCube depthSampler;
uniform float level;

void main(){
    color = textureLod(depthSampler, UVW, level).rgb;
}
