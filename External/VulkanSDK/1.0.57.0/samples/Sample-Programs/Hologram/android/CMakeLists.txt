#
# Copyright (C) 2017 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

project(Hologram)
cmake_minimum_required(VERSION 3.4.1)

get_filename_component(samplesDir "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
set(hologramDir "${samplesDir}/Sample-Programs/Hologram")
get_filename_component(glmDir "${samplesDir}/API-Samples/utils" ABSOLUTE)
get_filename_component(vulkanDir "${samplesDir}/include" ABSOLUTE)

# build native_app_glue as a static lib
add_library(native_activity_glue STATIC
            ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

# Build application's shared lib
set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -std=c++11  -fexceptions -Wall \
            -Wextra -Wno-unused-parameter \
            -DVK_NO_PROTOTYPES -DVK_USE_PLATFORM_ANDROID_KHR \
            -DGLM_FORCE_RADIANS")
add_library(Hologram SHARED
            ${hologramDir}/Shell.cpp
            ${hologramDir}/ShellAndroid.cpp
            ${hologramDir}/Simulation.cpp
            ${hologramDir}/Meshes.cpp
            ${hologramDir}/Hologram.cpp
            ${hologramDir}/Main.cpp
            ${CMAKE_SOURCE_DIR}/src/main/jni/HelpersDispatchTable.cpp)

target_include_directories(Hologram PRIVATE
            ${ANDROID_NDK}/sources/android/native_app_glue
            ${vulkanDir}
            ${glmDir}
            ${CMAKE_SOURCE_DIR}/src/main/jni)

target_link_libraries(Hologram
            android
            log
            native_activity_glue)
