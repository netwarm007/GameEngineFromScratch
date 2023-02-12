#!/bin/bash
set -e
git submodule update --init External/src/imgui
# source only module, only copy the backend source code
cp -v External/src/imgui/backends/imgui_impl_osx.* Platform/Darwin
cp -v External/src/imgui/backends/imgui_impl_android.* Platform/Android
cp -v External/src/imgui/backends/imgui_impl_sdl2.* Platform/Sdl
cp -v External/src/imgui/backends/imgui_impl_win32.* Platform/Windows
cp -v External/src/imgui/backends/imgui_impl_metal.* RHI/Metal
cp -v External/src/imgui/backends/imgui_impl_dx12.* RHI/D3d
cp -v External/src/imgui/backends/imgui_impl_opengl3.* RHI/OpenGL
cp -v External/src/imgui/backends/imgui_impl_opengl3_loader.* RHI/OpenGL
