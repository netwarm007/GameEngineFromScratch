add_executable(TextureCompressor TextureCompressor.cpp)

target_include_directories(TextureCompressor PRIVATE "${MYGE_EXTERNAL_ROOT}/include")
target_link_libraries(TextureCompressor Framework "${MYGE_EXTERNAL_ROOT}/lib/libispc_texcomp.a")