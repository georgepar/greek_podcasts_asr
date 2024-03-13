#!/usr/bin/env bash

input_rss=${1}
output_folder=${2}


# Included for more accurate reproduction. Corpus was downloaded at 12/2/2023, so include everything before
DATE_DOWNLOADED="12/2/2023"

mkdir -p $output_folder
cp $input_rss $output_folder
current_dir=$(pwd)

cd $output_folder
cat ${input_rss} | while read url
do
    npx podcast-dl \
        --episode-template "{{release_date}}-{{episode_num}}-{{guid}}" \
        --threads 8 \
        --include-episode-meta \
        --include-meta \
        --before ${DATE_DOWNLOADED} \
        --url ${url}
done
cd $current_dir
