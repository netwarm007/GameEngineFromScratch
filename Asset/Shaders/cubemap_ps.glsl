in vec3 UVW;

out vec3 color;

uniform samplerCube depthSampler;

void main(){
    color = texture(depthSampler, UVW).rgb;
}
