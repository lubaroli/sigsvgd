#!/bin/sh

BASEDIR="$(dirname "$0")"

input_folder="$1"

if [ ! -d "$input_folder" ]; then
  echo "The path '$input_folder' does not exists!"
  exit 1
fi

"$BASEDIR/img2gif.sh" $(ls "$input_folder"/*.png | sort -V) -o "$input_folder"/video.mp4 -- -framerate 10

