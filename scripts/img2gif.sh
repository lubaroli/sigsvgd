#!/bin/bash

# from
# https://gist.github.com/soraxas/d779c1cf7f4e9b74346d26c1cda8a6ff.git

#set -x
set -e

PWD="$(pwd)"
SCRATCH="$(mktemp -d )"

output_fname=out.gif

# cleanup after finish
function finish {
  rm -rf "$SCRATCH"
}
trap finish EXIT

function get_ext {
  _f="$(basename "$1")"
  echo "${_f##*.}"
}

function get_fname {
  _f="$(basename "$1")"
  echo "${_f%.*}"
}

function help {
  printf '%s\n' "Usage: $(basename $0) <IMG1 IMG2 ...> [-o|--output FNAME] [-- [FFMPEG_OPTS]] "
  printf '%s\n' ""
  printf '%s\n' "Every args (images path) before the -- will be used to create the gif, and"
  printf '%s\n' "every args after -- will be directly passed to ffmpeg."
  printf '%s\n' ""
  printf '%s\n' "Example"
  printf '%s\n' " >> $(basename $0) vis_1.png vis_2.png vis_3.png"
  printf '%s\n' " >> $(basename $0) \$(ls imgs/* | sort -V) -- -framerate 30"
  printf '%s\n' " >> $(basename $0) \$(ls imgs/* | sort -V) -o out.gif -- -r 5"
  exit
}



if test $# -eq 0; then
  help
fi

filelist=()

while test $# -gt 0; do
    case "$1" in
      -o|--output)
        shift
        output_fname="$1"
        ;;
      -h|--help)
        help
        ;;
      --verbose)
        export VERBOSE="true"
        ;;
      --)
        # the rest will be passed to ffmpeg
        breaknow=true
        ;;
      *)
        # set to file list
        filelist+=("$1")
        #set -- "$filelist" "$arg"
    esac
    shift
    if [ -n "$breaknow" ]; then
      break
    fi
done


ext="$(get_ext "${filelist[0]}")"
if [ "$(get_fname "${filelist[0]}")" = "$ext" ]; then
  echo "ERROR: cannot extract file extension for file '${filelist[0]}'"
  exit 1
fi

# sanity check
for file in "${filelist[@]}"; do
  # continue
  if [ "$ext" != "$(get_ext "$file")" ]; then
    echo "ERROR: Found at least two files with different extension. '${filelist[0]}' and '$file'"
    exit 1
  fi
done

##############################

# copy the target file to the scratch folder
_cnt=0
for file in "${filelist[@]}"; do
  _num=$(printf "%05d" "$_cnt") #04 pad to length of 4
  cp -- "$file" "$SCRATCH/$_num.$ext"
  let _cnt=_cnt+1
done


ffmpeg -framerate 15 $@ -i "$SCRATCH/%05d.$ext" "$output_fname"

