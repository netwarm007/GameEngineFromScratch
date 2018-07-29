#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 inputPosition;
layout(location = 1) in vec3 inputNormal;
layout(location = 2) in vec2 inputUV;

// update per frame
uniform mat4 depthVP;

// update per draw call
uniform mat4 modelMatrix;

void main(){
	gl_Position = depthVP * modelMatrix * vec4(inputPosition, 1.0f);
}
