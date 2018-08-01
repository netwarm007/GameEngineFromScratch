in vec2 UV;

out vec3 color;

uniform sampler2DArray depthSampler;
uniform float layer_index;

void main(){
    color = pow(texture(depthSampler, vec3(UV, layer_index)).rrr, vec3(10.0f));
}
