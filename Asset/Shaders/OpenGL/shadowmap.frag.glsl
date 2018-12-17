#version 420

layout(location = 0) out vec4 _entryPointOutput;

vec4 _shadowmap_frag_main()
{
    return vec4(1.0);
}

void main()
{
    _entryPointOutput = _shadowmap_frag_main();
}

