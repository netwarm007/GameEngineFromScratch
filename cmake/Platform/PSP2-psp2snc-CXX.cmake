include(Platform/PSP2-psp2snc)
__psp2_compiler_snc(CXX)
set (CMAKE_CXX_FLAGS                "-Xstd=cpp11")
set (CMAKE_CXX_FLAGS_DEBUG          "-g")
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

