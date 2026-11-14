git submodule update --init External/src/imgui
if errorlevel 1 exit /b 1
copy /Y External\src\imgui\backends\imgui_impl_osx.* Platform\Darwin\
if errorlevel 1 exit /b 1
copy /Y External\src\imgui\backends\imgui_impl_android.* Platform\Android\
if errorlevel 1 exit /b 1
copy /Y External\src\imgui\backends\imgui_impl_sdl2.* Platform\Sdl\
if errorlevel 1 exit /b 1
copy /Y External\src\imgui\backends\imgui_impl_win32.* Platform\Windows\
if errorlevel 1 exit /b 1
copy /Y External\src\imgui\backends\imgui_impl_metal.* RHI\Metal\
if errorlevel 1 exit /b 1
copy /Y External\src\imgui\backends\imgui_impl_dx12.* RHI\D3d\
if errorlevel 1 exit /b 1
copy /Y External\src\imgui\backends\imgui_impl_opengl3.* RHI\OpenGL\
if errorlevel 1 exit /b 1
copy /Y External\src\imgui\backends\imgui_impl_opengl3_loader.* RHI\OpenGL\
if errorlevel 1 exit /b 1