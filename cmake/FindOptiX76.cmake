# Looks for the environment variable:
# OPTIX76_PATH

# Sets the variables :
# OPTIX76_INCLUDE_DIR

# OptiX76_FOUND

set(OPTIX76_PATH $ENV{OPTIX76_PATH})

if ("${OPTIX76_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX76_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.6.0")
  else()
    # Adjust this if the OptiX SDK 7.6.0 installation is in a different location.
    set(OPTIX76_PATH "~/NVIDIA-OptiX-SDK-7.6.0-linux64")
  endif()
endif()

find_path(OPTIX76_INCLUDE_DIR optix_7_host.h ${OPTIX76_PATH}/include)

# message("OPTIX76_INCLUDE_DIR = " "${OPTIX76_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX76 DEFAULT_MSG OPTIX76_INCLUDE_DIR)

mark_as_advanced(OPTIX76_INCLUDE_DIR)

# message("OptiX76_FOUND = " "${OptiX76_FOUND}")