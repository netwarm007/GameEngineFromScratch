# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__PSP2_COMPILER_SNC)
  return()
endif()
set(__PSP2_COMPILER_SNC 1)

macro(__psp2_compiler_snc lang)
  # We pass this for historical reasons.  Projects may have
  # executables that use dlopen but do not set ENABLE_EXPORTS.
  set (CMAKE_${lang}_FLAGS_DEBUG          "-g")
  set (CMAKE_${lang}_FLAGS_RELWITHDEBINFO "-O2 -g")
endmacro()

