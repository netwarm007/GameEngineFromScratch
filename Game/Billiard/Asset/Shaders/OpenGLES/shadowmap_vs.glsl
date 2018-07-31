// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 inputPosition;

// update per frame
uniform mat4 depthVP;

void main(){
	gl_Position = depthVP * modelMatrix * vec4(inputPosition, 1.0f);
}
