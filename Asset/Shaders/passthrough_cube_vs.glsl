// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 inputPosition;
layout(location = 1) in vec3 inputUVW;

out vec3 UVW;

void main(){
	gl_Position = vec4(inputPosition, 1.0f);
	UVW = inputUVW;
}
