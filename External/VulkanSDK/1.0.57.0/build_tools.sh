#!/bin/bash
set -e

BINDIR="$PWD"/x86_64/bin
SHAREDDIR="$PWD"/x86_64/shared
LIBDIR="$PWD"/x86_64/lib
INCLUDEDIR="$PWD"/x86_64/include

buildVia() {
pushd source/via
cmake -DJSONCPP_SOURCE_DIR=./ -DJSONCPP_INCLUDE_DIR=./ -H. -Bbuild
make -Cbuild -j`nproc`
cp build/via "${BINDIR}"
popd
}

buildShaderc() {
pushd source/shaderc
python2 update_shaderc_sources.py
cd src
cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo -H. -Bbuild
sed -i -E 's:\/(.*?)build:..:g' build/libshaderc/shaderc_combined.ar
make -j`nproc` -Cbuild
cp build/glslc/glslc "${BINDIR}"
mkdir -p "${LIBDIR}"/libshaderc
ln -sf "$PWD"/build/libshaderc/libshaderc_combined.a "${LIBDIR}"/libshaderc
mkdir -p "${INCLUDEDIR}"/libshaderc
ln -sf "$PWD"/libshaderc/include/shaderc/shaderc.h "${INCLUDEDIR}"/libshaderc/
ln -sf "$PWD"/libshaderc/include/shaderc/shaderc.hpp "${INCLUDEDIR}"/libshaderc/
popd
}

buildSpirvTools() {
pushd source/spirv-tools
cp tools/emacs/50spirv-tools.el "${SHAREDDIR}"
cmake -H. -Bbuild 
make -Cbuild -j`nproc`
cp build/tools/spirv-as "${BINDIR}"
cp build/tools/spirv-cfg "${BINDIR}"
cp build/tools/spirv-dis "${BINDIR}"
cp build/tools/spirv-opt "${BINDIR}"
cp build/tools/spirv-val "${BINDIR}"
cd tools/lesspipe
chmod 755 spirv-lesspipe.sh
cp spirv-lesspipe.sh "${BINDIR}"
popd
}

buildSpirvCross() {
pushd source/spirv-cross
make -j`nproc`
cp spirv-cross "${BINDIR}"
popd
}

usage() {
    echo "Build tools script"
    echo "Usage: $CMDNAME [--via] [--shaderc] [--spirvtools] [--sprivcross]"
    echo ""
    echo "Omitting parameters will build every tool"
}

if [[ $# == 0 ]]; then
    buildVia
    buildShaderc
    buildSpirvTools
    buildSpirvCross
fi

# parse the arguments
while test $# -gt 0; do
        case "$1" in
	    --via)
		shift
		buildVia
		;;
	    --shaderc)
		shift
		buildShaderc
		;;
	    --spirvtools)
		shift
		buildSpirvTools
		;;
	    --spirvcross)
		shift
		buildSpirvCross
		;;
	    --help)
		usage
		;;
	    -h)
		usage
		;;
	    -)
		echo "error: unknown option"
		usage
		;;
	esac
	shift
done

			   
