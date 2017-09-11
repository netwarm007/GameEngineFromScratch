# - Find the ImageMagick binary suite.
# This module will search for a set of ImageMagick tools specified
# as components in the FIND_PACKAGE call. Typical components include,
# but are not limited to (future versions of ImageMagick might have
# additional components not listed here):
#
#  animate
#  compare
#  composite
#  conjure
#  convert
#  display
#  identify
#  import
#  mogrify
#  montage
#  stream
#
# If no component is specified in the FIND_PACKAGE call, then it only
# searches for the ImageMagick executable directory. This code defines
# the following variables:
#
#  ImageMagick_FOUND                  - TRUE if all components are found.
#  ImageMagick_EXECUTABLE_DIR         - Full path to executables directory.
#  ImageMagick_<component>_FOUND      - TRUE if <component> is found.
#  ImageMagick_<component>_EXECUTABLE - Full path to <component> executable.
#
# There are also components for the following ImageMagick APIs:
#
#  Magick++
#  MagickWand
#  MagickCore
#
# For these components the following variables are set:
#
#  ImageMagick_FOUND                    - TRUE if all components are found.
#  ImageMagick_INCLUDE_DIRS             - Full paths to all include dirs.
#  ImageMagick_LIBRARIES                - Full paths to all libraries.
#  ImageMagick_<component>_FOUND        - TRUE if <component> is found.
#  ImageMagick_<component>_INCLUDE_DIRS - Full path to <component> include dirs.
#  ImageMagick_<component>_LIBRARIES    - Full path to <component> libraries.
#
# Example Usages:
#  FIND_PACKAGE(ImageMagick)
#  FIND_PACKAGE(ImageMagick COMPONENTS convert)
#  FIND_PACKAGE(ImageMagick COMPONENTS convert mogrify display)
#  FIND_PACKAGE(ImageMagick COMPONENTS Magick++)
#  FIND_PACKAGE(ImageMagick COMPONENTS Magick++ convert)
#
# Note that the standard FIND_PACKAGE features are supported
# (i.e., QUIET, REQUIRED, etc.).

#=============================================================================
# Copyright 2007-2009 Kitware, Inc.
# Copyright 2007-2008 Miguel A. Figueroa-Villanueva <miguelf at ieee dot org>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright_cmake.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distributed this file outside of CMake, substitute the full
#  License text for the above reference.)

find_package(PkgConfig QUIET)

function(FIND_REGISTRY)
  if (WIN32)
  
    # If a 64-bit compile, it can only appear in "[HKLM]\\software\\ImageMagick"
    if (CMAKE_CL_64)

        GET_FILENAME_COMPONENT(IM_BIN_PATH  
          [HKEY_LOCAL_MACHINE\\SOFTWARE\\ImageMagick\\Current;BinPath]
          ABSOLUTE CACHE)
          
    else()
    
      # This is dumb, but it's the only way I've been able to get this to work.  CMake has no knowledge of the systems architecture.
      # So, if we want to detect if we're running a 32-bit compile on a 64-bit OS, we need to manually check for the existence of
      # ImageMagick in the WOW6432Node of the registry first.  If that fails, assume they want the 64-bit version.
      GET_FILENAME_COMPONENT(TESTING  
        [HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\ImageMagick\\Current;BinPath]
        PATH)

      # If the WOW6432Node reg string returns empty, assume 32-bit OS, and look in the standard reg path.
      if (TESTING STREQUAL "")
      
        GET_FILENAME_COMPONENT(IM_BIN_PATH  
          [HKEY_LOCAL_MACHINE\\SOFTWARE\\ImageMagick\\Current;BinPath]
          ABSOLUTE CACHE)

      # Otherwise, the WOW6432Node returned a string, assume 32-bit build on 64-bit OS and use that string.
      else()

        GET_FILENAME_COMPONENT(IM_BIN_PATH  
          [HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\ImageMagick\\Current;BinPath]
          ABSOLUTE CACHE)

      endif()

    endif()

    set (IMAGEMAGIC_REG_PATH ${IM_BIN_PATH} PARENT_SCOPE)
    set (IMAGEMAGIC_REGINCLUDE_PATH ${IM_BIN_PATH}/include PARENT_SCOPE)
    set (IMAGEMAGIC_REGLIB_PATH ${IM_BIN_PATH}/lib PARENT_SCOPE)

  else()

    # No registry exists for Linux.  So, just set these to empty strings.
    set (IMAGEMAGIC_REG_PATH "" PARENT_SCOPE)
    set (IMAGEMAGIC_REGINCLUDE_PATH "" PARENT_SCOPE)
    set (IMAGEMAGIC_REGLIB_PATH "" PARENT_SCOPE)

  endif()

endfunction()


#---------------------------------------------------------------------
# Helper functions
#---------------------------------------------------------------------
FUNCTION(FIND_IMAGEMAGICK_API component path header)
  SET(ImageMagick_${component}_FOUND FALSE PARENT_SCOPE)

  # NOTE: My experience is that Windows uses the PATHs
  #       variables, while Linux uses the PATH_SUFFIXES.
  #       You can't add sub-directories to the PATH_SUFFIXES
  #       because it messes up the ImageMagick Include
  #       paths that are returned.  Instead, you have to
  #       call this FIND_IMAGEMAGICK_API for each separate
  #       possible sub-folder.
  FIND_PATH(ImageMagick_${component}_INCLUDE_DIR
    NAMES ${header}
    PATHS
      ${ImageMagick_INCLUDE_DIRS}
      ${IMAGEMAGIC_REGINCLUDE_PATH}
    PATH_SUFFIXES
      ImageMagick ImageMagick-6 ImageMagick-7
    DOC "Path to the ImageMagick include dir."
    )
  FIND_PATH(ImageMagick_${component}_ARCH_INCLUDE_DIR
    NAMES magick/magick-baseconfig.h
    PATHS
      ${ImageMagick_INCLUDE_DIRS}
      ${IMAGEMAGIC_REGINCLUDE_PATH}
    PATH_SUFFIXES
      ImageMagick ImageMagick-6 ImageMagick-7
    DOC "Path to the ImageMagick arch-specific include dir."
    )
  FIND_LIBRARY(ImageMagick_${component}_LIBRARY
    NAMES ${ARGN}
    PATHS
      ${IMAGEMAGIC_REGLIB_PATH}
    DOC "Path to the ImageMagick Magick++ library."
    )


  IF (ImageMagick_${component}_INCLUDE_DIR AND ImageMagick_${component}_LIBRARY)
    SET(ImageMagick_${component}_FOUND TRUE PARENT_SCOPE)

    LIST(APPEND ImageMagick_INCLUDE_DIRS
      ${ImageMagick_${component}_INCLUDE_DIR}
      )

    # Only add the path if it's not the special string "<NONE>"
    IF(NOT path STREQUAL "<NONE>")
       LIST(APPEND ImageMagick_INCLUDE_DIRS
         ${ImageMagick_${component}_INCLUDE_DIR}/${path}
       )
    ENDIF()

    # Add the architecture include path if it exists
    IF (ImageMagick_${component}_ARCH_INCLUDE_DIR)
      LIST(APPEND ImageMagick_INCLUDE_DIRS
        ${ImageMagick_${component}_ARCH_INCLUDE_DIR}
        )
    ENDIF()

    LIST(REMOVE_DUPLICATES ImageMagick_INCLUDE_DIRS)
    SET(ImageMagick_INCLUDE_DIRS ${ImageMagick_INCLUDE_DIRS} PARENT_SCOPE)

    LIST(APPEND ImageMagick_LIBRARIES
      ${ImageMagick_${component}_LIBRARY}
      )
    SET(ImageMagick_LIBRARIES ${ImageMagick_LIBRARIES} PARENT_SCOPE)
  ENDIF()

ENDFUNCTION(FIND_IMAGEMAGICK_API)

FUNCTION(FIND_IMAGEMAGICK_EXE component)
  SET(_IMAGEMAGICK_EXECUTABLE
    ${ImageMagick_EXECUTABLE_DIR}/${component}${CMAKE_EXECUTABLE_SUFFIX})
  IF(EXISTS ${_IMAGEMAGICK_EXECUTABLE})
    SET(ImageMagick_${component}_EXECUTABLE
      ${_IMAGEMAGICK_EXECUTABLE}
       PARENT_SCOPE
       )
    SET(ImageMagick_${component}_FOUND TRUE PARENT_SCOPE)
  ELSE(EXISTS ${_IMAGEMAGICK_EXECUTABLE})
    SET(ImageMagick_${component}_FOUND FALSE PARENT_SCOPE)
  ENDIF(EXISTS ${_IMAGEMAGICK_EXECUTABLE})
ENDFUNCTION(FIND_IMAGEMAGICK_EXE)

#---------------------------------------------------------------------
# Start Actual Work
#---------------------------------------------------------------------
FIND_REGISTRY()

# Try to find a ImageMagick installation binary path.
FIND_PATH(ImageMagick_EXECUTABLE_DIR
  NAMES mogrify${CMAKE_EXECUTABLE_SUFFIX}
  PATHS
    ${IMAGEMAGIC_REG_PATH}
  DOC "Path to the ImageMagick binary directory."
  NO_DEFAULT_PATH
  )
FIND_PATH(ImageMagick_EXECUTABLE_DIR
  NAMES mogrify${CMAKE_EXECUTABLE_SUFFIX}
  )

# Find each component. Search for all tools in same dir
# <ImageMagick_EXECUTABLE_DIR>; otherwise they should be found
# independently and not in a cohesive module such as this one.
SET(ImageMagick_FOUND TRUE)
FOREACH(component ${ImageMagick_FIND_COMPONENTS}
    # DEPRECATED: forced components for backward compatibility
    convert mogrify import montage composite
    )

    IF(component STREQUAL "Magick++")
        # unset cached variable that assumes header parameter never changes
        UNSET(ImageMagick_MagickWand_INCLUDE_DIR CACHE)

        # Try top folder first
        FIND_IMAGEMAGICK_API(Magick++ <NONE> Magick++.h
            Magick++ CORE_RL_Magick++_ Magick++-6.Q16 Magick++-Q16 Magick++-6.Q8 Magick++-Q8 Magick++-6.Q16HDRI Magick++-Q16HDRI Magick++-6.Q8HDRI Magick++-Q8HDRI
        )

        IF(NOT ImageMagick_Magick++_INCLUDE_DIR OR NOT ImageMagick_Magick++_LIBRARY)
            # Look for Magick++.h, in the magick++ sub-folder
            FIND_IMAGEMAGICK_API(Magick++ magick++ magick++/Magick++.h
                Magick++ CORE_RL_Magick++_ Magick++-6.Q16 Magick++-Q16 Magick++-6.Q8 Magick++-Q8 Magick++-6.Q16HDRI Magick++-Q16HDRI Magick++-6.Q8HDRI Magick++-Q8HDRI
            )
        ENDIF()

        IF(NOT ImageMagick_Magick++_INCLUDE_DIR OR NOT ImageMagick_Magick++_LIBRARY)
            # Look for Magick++.h, in the magick sub-folder
            FIND_IMAGEMAGICK_API(Magick++ magick magick/Magick++.h
                Magick++ CORE_RL_Magick++_ Magick++-6.Q16 Magick++-Q16 Magick++-6.Q8 Magick++-Q8 Magick++-6.Q16HDRI Magick++-Q16HDRI Magick++-6.Q8HDRI Magick++-Q8HDRI
            )
        ENDIF()

    ELSEIF(component STREQUAL "MagickWand")
        # unset cached variable that assumes header parameter never changes
        UNSET(ImageMagick_MagickWand_INCLUDE_DIR CACHE)

        # Try top folder first
        FIND_IMAGEMAGICK_API(MagickWand <NONE> MagickWand.h
           Wand MagickWand CORE_RL_wand_ CORE_RL_MagickWand_ MagickWand-6.Q16 MagickWand-Q16 MagickWand-6.Q8 MagickWand-Q8 MagickWand-6.Q16HDRI MagickWand-Q16HDRI MagickWand-6.Q8HDRI MagickWand-Q8HDRI
        )

        IF(NOT ImageMagick_MagickWand_INCLUDE_DIR OR NOT ImageMagick_MagickWand_LIBRARY)
            # Look for MagickWand.h in the wand sub-folder
            FIND_IMAGEMAGICK_API(MagickWand wand wand/MagickWand.h
              Wand MagickWand CORE_RL_wand_ CORE_RL_MagickWand_ MagickWand-6.Q16 MagickWand-Q16 MagickWand-6.Q8 MagickWand-Q8 MagickWand-6.Q16HDRI MagickWand-Q16HDRI MagickWand-6.Q8HDRI MagickWand-Q8HDRI
            )
        ENDIF()

        IF(NOT ImageMagick_MagickWand_INCLUDE_DIR OR NOT ImageMagick_MagickWand_LIBRARY)
            # Look for MagickWand.h he MagickWand sub-folder
            FIND_IMAGEMAGICK_API(MagickWand MagickWand MagickWand/MagickWand.h
              Wand MagickWand CORE_RL_wand_ CORE_RL_MagickWand_ MagickWand-6.Q16 MagickWand-Q16 MagickWand-6.Q8 MagickWand-Q8 MagickWand-6.Q16HDRI MagickWand-Q16HDRI MagickWand-6.Q8HDRI MagickWand-Q8HDRI
            )
        ENDIF()

        IF(NOT ImageMagick_MagickWand_INCLUDE_DIR OR NOT ImageMagick_MagickWand_LIBRARY)
            # Look for MagickWand.h he magick sub-folder
            FIND_IMAGEMAGICK_API(MagickWand magick magick/MagickWand.h
              Wand MagickWand CORE_RL_wand_ CORE_RL_MagickWand_ MagickWand-6.Q16 MagickWand-Q16 MagickWand-6.Q8 MagickWand-Q8 MagickWand-6.Q16HDRI MagickWand-Q16HDRI MagickWand-6.Q8HDRI MagickWand-Q8HDRI
            )
        ENDIF()

    ELSEIF(component STREQUAL "MagickCore")

        # Try top folder first
        FIND_IMAGEMAGICK_API(MagickCore <NONE> MagickCore.h
          Magick MagickCore CORE_RL_magick_ CORE_RL_MagickCore_ MagickCore-6.Q16 MagickCore-Q16 MagickCore-6.Q8 MagickCore-Q8 MagickCore-6.Q16HDRI MagickCore-Q16HDRI MagickCore-6.Q8HDRI MagickCore-Q8HDRI
        )

        IF(NOT ImageMagick_MagickCore_INCLUDE_DIR OR NOT ImageMagick_MagickCore_LIBRARY)
            # Look for MagickCore.h, in the MagickCore sub-folder
            FIND_IMAGEMAGICK_API(MagickCore MagickCore MagickCore/MagickCore.h
              Magick MagickCore CORE_RL_magick_ CORE_RL_MagickCore_ MagickCore-6.Q16 MagickCore-Q16 MagickCore-6.Q8 MagickCore-Q8 MagickCore-6.Q16HDRI MagickCore-Q16HDRI MagickCore-6.Q8HDRI MagickCore-Q8HDRI
            )
        ENDIF()

        IF(NOT ImageMagick_MagickCore_INCLUDE_DIR OR NOT ImageMagick_MagickCore_LIBRARY)
            # Look for MagickCore.h, in the magick sub-folder
            FIND_IMAGEMAGICK_API(MagickCore magick magick/MagickCore.h
              Magick MagickCore CORE_RL_magick_ CORE_RL_MagickCore_ MagickCore-6.Q16 MagickCore-Q16 MagickCore-6.Q8 MagickCore-Q8 MagickCore-6.Q16HDRI MagickCore-Q16HDRI MagickCore-6.Q8HDRI MagickCore-Q8HDRI
            )
        ENDIF()

    ENDIF()
  
    IF(NOT ImageMagick_${component}_FOUND)

        LIST(FIND ImageMagick_FIND_COMPONENTS ${component} is_requested)
        IF(is_requested GREATER -1)
            SET(ImageMagick_FOUND FALSE)
        ENDIF(is_requested GREATER -1)

    ENDIF(NOT ImageMagick_${component}_FOUND)
ENDFOREACH(component)

#---------------------------------------------------------------------
# Standard Package Output
#---------------------------------------------------------------------
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
  ImageMagick DEFAULT_MSG ImageMagick_FOUND
  )
# Maintain consistency with all other variables.
SET(ImageMagick_FOUND ${IMAGEMAGICK_FOUND})

#---------------------------------------------------------------------
# DEPRECATED: Setting variables for backward compatibility.
#---------------------------------------------------------------------
SET(IMAGEMAGICK_BINARY_PATH          ${ImageMagick_EXECUTABLE_DIR}
    CACHE PATH "Path to the ImageMagick binary directory.")
SET(IMAGEMAGICK_CONVERT_EXECUTABLE   ${ImageMagick_convert_EXECUTABLE}
    CACHE FILEPATH "Path to ImageMagick's convert executable.")
SET(IMAGEMAGICK_MOGRIFY_EXECUTABLE   ${ImageMagick_mogrify_EXECUTABLE}
    CACHE FILEPATH "Path to ImageMagick's mogrify executable.")
SET(IMAGEMAGICK_IMPORT_EXECUTABLE    ${ImageMagick_import_EXECUTABLE}
    CACHE FILEPATH "Path to ImageMagick's import executable.")
SET(IMAGEMAGICK_MONTAGE_EXECUTABLE   ${ImageMagick_montage_EXECUTABLE}
    CACHE FILEPATH "Path to ImageMagick's montage executable.")
SET(IMAGEMAGICK_COMPOSITE_EXECUTABLE ${ImageMagick_composite_EXECUTABLE}
    CACHE FILEPATH "Path to ImageMagick's composite executable.")

MARK_AS_ADVANCED(
  IMAGEMAGICK_BINARY_PATH
  IMAGEMAGICK_CONVERT_EXECUTABLE
  IMAGEMAGICK_MOGRIFY_EXECUTABLE
  IMAGEMAGICK_IMPORT_EXECUTABLE
  IMAGEMAGICK_MONTAGE_EXECUTABLE
  IMAGEMAGICK_COMPOSITE_EXECUTABLE
  )
