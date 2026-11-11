#ifndef MYGE_ROOT_SIGNATURE_H_HLSL
#define MYGE_ROOT_SIGNATURE_H_HLSL

#ifdef MYGE_GLSLANG
#define MYGE_ROOT_SIGNATURE(signature)
#else
#define MYGE_ROOT_SIGNATURE(signature) [RootSignature(signature)]
#endif

#endif
