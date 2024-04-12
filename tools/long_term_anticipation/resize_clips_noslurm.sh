#!/bin/bash

source ~/.bashrc
cd /proj/vondrick/didac/code/forecasting

EGO4D_CLIP_DIR=data/long_term_anticipation/clips_hq/
TARGET_DIR=data/long_term_anticipation/clips/
mkdir -p ${TARGET_DIR}

readarray -d '' CLIPS < <(find ${EGO4D_CLIP_DIR} -name "*.mp4" -print0)
IFS=$'\n' CLIPS=($(sort <<<"${CLIPS[*]}"))
unset IFS

#echo "${CLIPS[@]}" | parallel -j5 'echo {} && sleep 1'


# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

N=50
open_sem $N
for CLIP in "${CLIPS[@]}"
do
  FILENAME=`basename $CLIP`
  echo $FILENAME

  run_with_lock ffmpeg -y -i $CLIP \
    -c:v libx264 \
    -crf 28 \
    -vf "scale=320:320:force_original_aspect_ratio=increase,pad='iw+mod(iw,2)':'ih+mod(ih,2)'" \
    -an ${TARGET_DIR}/${FILENAME}
done