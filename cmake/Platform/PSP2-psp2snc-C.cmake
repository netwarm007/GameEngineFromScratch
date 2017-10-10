include(Platform/PSP2-psp2snc)
__psp2_compiler_snc(C)
set (CMAKE_C_FLAGS                  "-Xstd=c99")
set (CMAKE_C_FLAGS_DEBUG            "-g")
set (CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g")

