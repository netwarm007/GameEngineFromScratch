in vec3 UVW;

out vec3 color;

uniform samplerCubeArray depthSampler;
uniform float layer_index;
uniform float level;

void main(){
    color = textureLod(depthSampler, vec4(UVW, layer_index), level).rrr;
}
