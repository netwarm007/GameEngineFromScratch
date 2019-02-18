#version 300 es
precision mediump float;
precision highp int;

layout(location = 0) out highp vec4 _entryPointOutput;

highp vec4 _shadowmap_frag_main()
{
    return vec4(1.0);
}

void main()
{
    _entryPointOutput = _shadowmap_frag_main();
}

