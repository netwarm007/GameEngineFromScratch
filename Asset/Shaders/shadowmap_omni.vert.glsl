// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 inputPosition;

void main(){
	gl_Position = modelMatrix * vec4(inputPosition, 1.0f);
}
