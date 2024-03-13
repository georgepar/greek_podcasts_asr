#!/usr/bin/env bash

INPUT_FOLDER=$(pwd)/rss-lists/tts
OUTPUT_FOLDER=tts-candidate-data

mkdir -p ${OUTPUT_FOLDER}

for rss in $(ls ${INPUT_FOLDER}/*.txt)
do
    bash download_individual_rss_list.sh $rss ${OUTPUT_FOLDER}
done
