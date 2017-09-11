#!/bin/bash
SAVEIMAGES=""
COMPAREIMAGES=""
ARGS=""
RED='\e[0;31m'
GREEN='\e[0;32m'
NOCOLOR='\e[0m'
RETVAL=0

# run all of the samples
# parse the arguments
while test $# -gt 0; do
        case "$1" in
        --save-images)
                SAVEIMAGES="true"
		ARGS="--save-images $ARGS"
                ;;
        --compare-images)
                COMPAREIMAGES="true"
		ARGS="--save-images $ARGS"
                ;;
        -)
                echo "error: unknown option"
                usage
                ;;
        esac
        shift
done

if ! test -z $COMPAREIMAGES; then
    if ! test -d 'golden'; then 
        { echo >&2 "I require golden images but no golden directory.  Aborting."; exit 1; }
    fi
    type compare >/dev/null 2>&1 || { echo >&2 "I require ImageMagick compare but it's not installed.  Aborting."; exit 1; }
fi

#  save command directory; note that this file is a link to file of same
# name in samples/src/
CMDDIR="$(dirname "$(readlink -f ${BASH_SOURCE[0]})")"

# get the list of built samples to run
SAMP2RUN=`find "$CMDDIR" -name *.cpp -not -path "*util*" -not -path "*android*" -not -path "*CMakeFiles*" | sort`
#echo "SAMP2RUN is $SAMP2RUN"

# display short description of the sample and run it
IFS=$(echo -en "\n\b")
for f in $SAMP2RUN
do
   # get short description of the sample source file
   DESCRIPT=`"$CMDDIR/get-short-descripts.sh" --nofname --sampfname "$f"`
   BNAME=$(basename $f)
   echo "RUNNING SAMPLE:  $BNAME"
   echo "  ** $DESCRIPT"

   # run the built sample; need to remove .cpp from name
   RNAME=./${BNAME%.cpp}
   $RNAME $ARGS
   echo ""
   if ! test -z $COMPAREIMAGES; then
       GOLDNAME="golden/${RNAME}.ppm"
        if test -f $GOLDNAME; then
           THISNAME="${RNAME}.ppm"
	   CMDRES=`compare -metric AE -fuzz 3% $THISNAME $GOLDNAME ${RNAME}-diff.ppm 2>&1` 
           if [ $CMDRES == "0" ]; then
               >&2 echo -e "${GREEN}${RNAME} PASS${NOCOLOR}"
           else
               >&2 echo -e "${RED}${RNAME} FAIL${NOCOLOR} : pixel error count is ${CMDRES}"
               $SAVEIMAGES="true"
	       RETVAL=1
           fi
        else
            if test -f ${RNAME}.ppm; then
                >&2 echo -e "${RED}${RNAME} FAIL${NOCOLOR} : Missing Golden Image"
                RETVAL=1
            fi
        fi
    fi
    if test -z $SAVEIMAGES; then
        `rm ${RNAME}.ppm > /dev/null 2>&1`
	`rm ${RNAME}-diff.ppm > /dev/null 2>&1`
    fi
done
exit $RETVAL

