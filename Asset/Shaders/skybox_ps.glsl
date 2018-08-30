// Ouput data
layout(location = 0) out vec4 Color;

in vec3 UVW;

uniform samplerCube skybox;

void main(){
    Color = textureLod(skybox, UVW, 2);
}
