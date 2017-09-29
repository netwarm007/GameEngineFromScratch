# this one is important
SET(CMAKE_SYSTEM_NAME Linux)
#this one not so much
SET(CMAKE_SYSTEM_VERSION 3.570)

SET(SNC_TOOLCHAIN_DIR "$ENV{SCE_PSP2_SDK_DIR}/host_tools/build/bin/")
SET(SNC_TARGET_ROOT_DIR "$ENV{SCE_PSP2_SDK_DIR}/target")

# specify the cross compiler
SET(CMAKE_C_COMPILER ${SNC_TOOLCHAIN_DIR}psp2snc.exe)
SET(CMAKE_CXX_COMPILER ${SNC_TOOLCHAIN_DIR}psp2snc.exe)
SET(CMAKE_AR ${SNC_TOOLCHAIN_DIR}psp2snarl.exe CACHE INTERNAL "Archieve")

# where is the target environment 
SET(CMAKE_FIND_ROOT_PATH  ${SNC_TARGET_ROOT_DIR})

# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
