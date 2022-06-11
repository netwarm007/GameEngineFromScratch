add_executable(TextureCompressor TextureCompressor.cpp)

target_link_libraries(TextureCompressor Framework "${MYGE_EXTERNAL_ROOT}/lib/libispc_texcomp.a")

add_executable(MaterialBaker MaterialBaker.cpp)
target_link_libraries(MaterialBaker Framework "${MYGE_EXTERNAL_ROOT}/lib/libispc_texcomp.a")