#!/usr/bin/bash

# Runs segmentation of a kaldi dir and saves the segmented wavs to the output wav folder
# Requires the utils/data/extract_wav_segments_data_dir.sh script from the wsj/s5 recipe
set -xe

NUM_JOBS=24
KALDI_DIR_IN=$1
KALDI_DIR_OUT=$2
WAV_FOLDER=$3

INTERMEDIATE_DIR=${KALDI_DIR_IN}_segments_lazy

bash utils/data/extract_wav_segments_data_dir.sh --nj $NUM_JOBS $KALDI_DIR_IN $INTERMEDIATE_DIR
python kaldipathlib.py execute-wav-scp $INTERMEDIATE_DIR $KALDI_DIR_OUT $WAV_FOLDER
