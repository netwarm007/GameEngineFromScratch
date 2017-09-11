#!/bin/bash

# get long description of vulkan sample applications
#  - currently displays descriptions for all samples that have set
#    a long description via the VULKAN_SAMPLE_DESCRIPTION_START and
#    VULKAN_SAMPLE_DESCRIPTION_END keywords
#
# usage:  get-descripts.sh
#
# TODO - support filename(s) options - display descripts for only those samples

# save command directory
CMDDIR="$(dirname "$(readlink -f ${BASH_SOURCE[0]})")"

SAMPS=`find "$CMDDIR" -name *.cpp -not -path "*util*"`

# read all source .cpp files and display the long description
IFS=$(echo -en "\n\b")
for f in $SAMPS
do
   DESCRIPT=`sed -n '/^VULKAN_SAMPLE_DESCRIPTION_START$/,/^VULKAN_SAMPLE_DESCRIPTION_END$/{ /^VULKAN_SAMPLE_DESCRIPTION_START/d; /^VULKAN_SAMPLE_DESCRIPTION_END/d; p; }' "$f"`
   BNAME=`basename $f`
   if [ ! -z "$DESCRIPT" ]; then
       echo "$BNAME:"
       echo "$DESCRIPT"
       echo ""
   else
       echo "$BNAME:"
       echo No Description
       echo ""
   fi
done

