# this one is important
SET(CMAKE_SYSTEM_NAME ORBIS)
#this one not so much
SET(CMAKE_SYSTEM_VERSION 5.008)

STRING(REGEX REPLACE "\\\\" "/" SCE_ORBIS_SDK_DIR $ENV{SCE_ORBIS_SDK_DIR})
SET(ORBIS_TOOLCHAIN_DIR "${SCE_ORBIS_SDK_DIR}/host_tools/bin/")
SET(ORBIS_TARGET_ROOT_DIR "${SCE_ORBIS_SDK_DIR}/target")

# specify the cross compiler
SET(CMAKE_C_COMPILER ${ORBIS_TOOLCHAIN_DIR}orbis-clang.exe)
SET(CMAKE_CXX_COMPILER ${ORBIS_TOOLCHAIN_DIR}orbis-clang.exe)
SET(CMAKE_AR ${ORBIS_TOOLCHAIN_DIR}orbis-ar.exe CACHE INTERNAL "Archieve")

# where is the target environment 
SET(CMAKE_FIND_ROOT_PATH  ${ORBIS_TARGET_ROOT_DIR})

# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
