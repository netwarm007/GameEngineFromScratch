#!/bin/bash

# return short description of identified or all samples
#    get-short-descript.sh [--nofname] [--sampfname sample_source_file_name]
#
# - uses VULKAN_SAMPLE_SHORT_DESCRIPTION keyword from source file
# - default (no options) will display short descriptions for all samples
# - default display format: displays sample file names and short descriptions
#   filename.cpp:  short descript
# - specify --nofname to suppress display of the filename

NOFNAME=""

# save command directory
CMDDIR="$(dirname "$(readlink -f ${BASH_SOURCE[0]})")"

# default to displaying short descriptions of all samples
SAMP2DISP=`find "$CMDDIR" -name *.cpp -not -path "*util*"`


# parse the arguments
while test $# -gt 0; do
        case "$1" in
        --nofname)
                NOFNAME=true
                ;;
        --sampfname)
                shift
                SAMP2DISP=$1
                ;;
        esac

        shift
done

# read all identified .cpp file(s) and display the short description
IFS=$(echo -en "\n\b")
for f in $SAMP2DISP
do
   SHORT_DESCRIPT=`sed -n '/VULKAN_SAMPLE_SHORT_DESCRIPTION/{n;p}' $f`
   BNAME=`basename $f`
   if test -z $NOFNAME; then
      echo "$BNAME: $SHORT_DESCRIPT"
   else
      echo "$SHORT_DESCRIPT"
   fi
done

