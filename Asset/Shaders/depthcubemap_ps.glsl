in vec3 UVW;

out vec3 color;

uniform samplerCubeArray depthSampler;
uniform float layer_index;

void main(){
    color = texture(depthSampler, vec4(UVW, layer_index)).rrr;
}
