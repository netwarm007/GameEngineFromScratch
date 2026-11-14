add_library(GeomMath
CrossProduct.ispc 
DotProduct.ispc
MulByElement.ispc
Transpose.ispc
Normalize.ispc
Transform.ispc
AddByElement.ispc
SubByElement.ispc
MatrixUtil.ispc
InverseMatrix.ispc
DCT.ispc
Absolute.ispc
Pow.ispc 
DivByElement.ispc
Sqrt.ispc
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set_property(TARGET GeomMath PROPERTY ISPC_INSTRUCTION_SETS neon-i32x4)
    target_compile_options(GeomMath PRIVATE --arch=aarch64)
endif()
