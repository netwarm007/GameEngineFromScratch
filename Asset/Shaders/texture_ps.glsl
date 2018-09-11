in vec2 UV;

out vec3 color;

uniform sampler2D tex;

void main(){
    color = texture(tex, UV).rgb;
}
