layout(location = 0) in vec2 UV;

layout(location = 0) out vec3 color;

layout(push_constant) uniform debugPushConstants {
    float layer_index;
} u_pushConstants;

layout(binding = 0) uniform sampler2DArray depthSampler;

void main(){
    color = texture(depthSampler, vec3(UV, u_pushConstants.layer_index)).rrr;
}
