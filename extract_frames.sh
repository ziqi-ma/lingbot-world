#!/bin/bash
VIDEO=$1
OUTPUT_DIR="${2:-${VIDEO%.*}_frames}"

mkdir -p "$OUTPUT_DIR"
ffmpeg -i "$VIDEO" "$OUTPUT_DIR/frame_%04d.png"
echo "Frames saved to $OUTPUT_DIR"
