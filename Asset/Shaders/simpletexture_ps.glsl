#version 330 core

in vec2 UV;

out vec3 color;

uniform sampler2D renderedTexture;

void main(){
    color = texture(renderedTexture, UV).rgb;
}
