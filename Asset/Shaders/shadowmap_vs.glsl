#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 inputPosition;

// update per frame
uniform mat4 depthMVP;

// update per draw call
uniform mat4 modelMatrix;

void main(){
	gl_Position = depthMVP * modelMatrix * vec4(inputPosition, 1.0f);
}
