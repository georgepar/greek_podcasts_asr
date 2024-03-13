#!/usr/bin/env bash
set -x

INPUT_FOLDER=${1}
CURRENT_FOLDER=$(pwd)
DOMAIN_FOLDERS=$(find ${INPUT_FOLDER}  -mindepth 1 -maxdepth 1 -type d)
export HF_TOKEN=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
export WHISPER_MODEL=large-v3

function transcribe_wav() {
    input_wav="${1}"
    output_dir="${2}"
    echo "Transcribing ${input_wav} in ${output_dir}"
    whisperx --diarize --print_progress True --verbose True --model ${WHISPER_MODEL} --hf_token ${HF_TOKEN} --threads 24 --device cuda --output_dir ${output_dir} "${input_wav}"
}

export -f transcribe_wav

function transcribe_folder() {
    domain="${1}"
    find $domain -name "*.wav" | parallel -j3 whisperx --language el --diarize --print_progress True --verbose True --model ${WHISPER_MODEL} --hf_token ${HF_TOKEN} --threads 2 --device cuda --output_dir ${domain} "{}"

}

export -f transcribe_folder

# for domain in $DOMAIN_FOLDERS
# do
#     find $domain -name "*.wav" | parallel -j2 whisperx --language el --diarize --print_progress True --verbose True --model ${WHISPER_MODEL} --hf_token ${HF_TOKEN} --threads 24 --device cuda --output_dir ${domain} "{}"


#     # for f in $(find $domain -name "*.wav")
#     # do
#     #     transcribe_wav "$f" "$domain"
#     # done
# done

 find ${INPUT_FOLDER}  -mindepth 1 -maxdepth 1 -type d | parallel -j2 CUDA_VISIBLE_DEVICES='$(({%} - 1))' transcribe_folder {}

