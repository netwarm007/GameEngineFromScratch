# Utility Functions for the Vulkan Samples Kit

## utils.hpp/utils.cpp

- get_base_data_dir() - return full path to the base data directory; uses
  CMAKE definition VULKAN_SAMPLES_BASE_DIR to get the samples base directory
  and appends “/data/”
- get_data_dir(__FILE__) - return the full path to the release-specific data
  directory
  - the version prefix is extracted from __FILE__ to determine the release
    specific data directory component
  - implies that function must be called from the main sample source file,
    not another utility

Other utility functions may be added to utils.cpp, or new source files created.

