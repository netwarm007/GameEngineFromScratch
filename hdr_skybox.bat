REM HDR to DDS
External\Windows\bin\cmft_cli.exe --useOpenCL true --inputFacePosX Asset\Textures\hdr\%1_posx.hdr --inputFaceNegX Asset\Textures\hdr\%1_negx.hdr --inputFacePosY Asset\Textures\hdr\%1_posy.hdr --inputFaceNegY Asset\Textures\hdr\%1_negy.hdr --inputFacePosZ Asset\Textures\hdr\%1_posz.hdr --inputFaceNegZ Asset\Textures\hdr\%1_negz.hdr --filter none --outputNum 1 --output0 Asset\Textures\hdr\%1 --output0params dds,rgba16f,facelist

REM irradiance
External\Windows\bin\cmft_cli.exe --useOpenCL true --inputFacePosX Asset\Textures\hdr\%1_posx.hdr --inputFaceNegX Asset\Textures\hdr\%1_negx.hdr --inputFacePosY Asset\Textures\hdr\%1_posy.hdr --inputFaceNegY Asset\Textures\hdr\%1_negy.hdr --inputFacePosZ Asset\Textures\hdr\%1_posz.hdr --inputFaceNegZ Asset\Textures\hdr\%1_negz.hdr --filter irradiance --outputNum 1 --output0 Asset\Textures\hdr\%1_irradiance --output0params dds,rgba16f,facelist --dstFaceSize 512

REM radiance
External\Windows\bin\cmft_cli.exe --numCpuProcessingThreads 6 --useOpenCL true --inputFacePosX Asset\Textures\hdr\%1_posx.hdr --inputFaceNegX Asset\Textures\hdr\%1_negx.hdr --inputFacePosY Asset\Textures\hdr\%1_posy.hdr --inputFaceNegY Asset\Textures\hdr\%1_negy.hdr --inputFacePosZ Asset\Textures\hdr\%1_posz.hdr --inputFaceNegZ Asset\Textures\hdr\%1_negz.hdr --filter radiance --excludeBase false --mipCount 9 --glossScale 10 --glossBias 1 --lightingModel phongbrdf --outputNum 1 --output0 Asset\Textures\hdr\%1_radiance --output0params dds,rgba16f,facelist
