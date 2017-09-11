#!/bin/bash
OUTDIR="${OUTDIR:-out}"
mkdir -p "${OUTDIR}/html"
mkdir -p "${OUTDIR}/css"
mkdir -p "${OUTDIR}/images"

for i in markdown/*.md; do
    file=${i##*/}
    base=${file%.*}
    perl tools/Markdown.pl --html4tags $i > "${OUTDIR}/html/${base}.html"
done
cp css/lg_stylesheet.css "${OUTDIR}/css"
cp images/*.png "${OUTDIR}/images"
cp images/bg-starfield.jpg "${OUTDIR}/images"
