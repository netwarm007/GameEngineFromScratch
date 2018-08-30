// Ouput data
layout(location = 0) out vec4 outputColor;

in vec3 UVW;

uniform samplerCube skybox;

void main(){
    outputColor = textureLod(skybox, UVW, 2);

    // inverse gamma correction
    outputColor.rgb = inverse_gamma_correction(outputColor.rgb);

    // tone mapping
    //outputColor.rgb = reinhard_tone_mapping(outputColor.rgb);
    outputColor.rgb = exposure_tone_mapping(outputColor.rgb);

    // gamma correction
    outputColor.rgb = gamma_correction(outputColor.rgb);
}
