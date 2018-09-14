layout(location = 0) in vec2 UV;
layout(location = 0) out vec3 color;

layout(binding = 0) uniform sampler2D tex;

void main(){
    color = texture(tex, UV).rgb;
}
