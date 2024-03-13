
mkdir -p results_hplg
CUDA_VISIBLE_DEVICES=0 python decode_whisper_hplg.py --processor openai/whisper-small --model openai/whisper-small --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_small_openai &
CUDA_VISIBLE_DEVICES=1 python decode_whisper_hplg.py --processor openai/whisper-small --model openai/whisper-small --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_small_openai &
CUDA_VISIBLE_DEVICES=2 python decode_whisper_hplg.py --processor openai/whisper-medium --model openai/whisper-medium --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_medium_openai  &
CUDA_VISIBLE_DEVICES=3 python decode_whisper_hplg.py --processor openai/whisper-medium --model openai/whisper-medium --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_medium_openai

CUDA_VISIBLE_DEVICES=0 python decode_whisper_hplg.py --processor whisper-small-el-2h-hf/  --model whisper-small-el-2h-hf/checkpoint-1000 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_small_2h  &
CUDA_VISIBLE_DEVICES=1 python decode_whisper_hplg.py --processor whisper-small-el-2h-hf/  --model whisper-small-el-2h-hf/checkpoint-1000 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_small_2h  &
CUDA_VISIBLE_DEVICES=2 python decode_whisper_hplg.py --processor whisper-medium-el-2h-hf/  --model whisper-medium-el-2h-hf/checkpoint-1000 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_medium_2h  &
CUDA_VISIBLE_DEVICES=3 python decode_whisper_hplg.py --processor whisper-medium-el-2h-hf/  --model whisper-medium-el-2h-hf/checkpoint-1000 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_medium_2h
wait

CUDA_VISIBLE_DEVICES=0 python decode_whisper_hplg.py --processor whisper-small-el-5h-hf/  --model whisper-small-el-5h-hf/checkpoint-2375 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_small_5h  &
CUDA_VISIBLE_DEVICES=1 python decode_whisper_hplg.py --processor whisper-small-el-5h-hf/  --model whisper-small-el-5h-hf/checkpoint-2375 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_small_5h  &
CUDA_VISIBLE_DEVICES=2 python decode_whisper_hplg.py --processor whisper-medium-el-5h-hf/  --model whisper-medium-el-5h-hf/checkpoint-2375 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_medium_5h  &
CUDA_VISIBLE_DEVICES=3 python decode_whisper_hplg.py --processor whisper-medium-el-5h-hf/  --model whisper-medium-el-5h-hf/checkpoint-2375 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_medium_5h &
wait

CUDA_VISIBLE_DEVICES=0 python decode_whisper_hplg.py --processor openai/whisper-large-v2 --model openai/whisper-large-v2 --text-key transcription --batch-size 16 --dataset logotypografia > results_hplg/logotypografia_largev2_openai &
CUDA_VISIBLE_DEVICES=1 python decode_whisper_hplg.py --processor openai/whisper-large-v3 --model openai/whisper-large-v3 --text-key transcription --batch-size 16 --dataset logotypografia > results_hplg/logotypografia_largev3_openai  &
CUDA_VISIBLE_DEVICES=2 python decode_whisper_hplg.py --processor openai/whisper-large-v2 --model openai/whisper-large-v2 --text-key transcription --batch-size 16 --dataset hparl > results_hplg/hparl_largev2_openai &
CUDA_VISIBLE_DEVICES=3 python decode_whisper_hplg.py --processor openai/whisper-large-v3 --model openai/whisper-large-v3 --text-key transcription --batch-size 16 --dataset hparl > results_hplg/hparl_largev3_openai &
wait
# python decode_whisper_cv.py --processor whisper-medium-el-20h-hf/ --model whisper-medium-el-20h-hf/checkpoint-10000 --text-key sentence --dataset mozilla-foundation/common_voice_11_0 --lang el > results/medium_20h_cv
# python decode_whisper_cv.py --processor whisper-medium-el-20h-hf/ --model whisper-medium-el-20h-hf/checkpoint-10000 --text-key transcription --dataset google/fleurs --lang el_gr > results/medium_20h_fleurs
# python decode_whisper_cv.py --processor whisper-small-el-20h-hf/ --model whisper-small-el-20h-hf/checkpoint-10000 --text-key sentence --dataset mozilla-foundation/common_voice_11_0 --lang el > results/small_20h_cv
# python decode_whisper_cv.py --processor whisper-small-el-20h-hf/ --model whisper-small-el-20h-hf/checkpoint-10000 --text-key transcription --dataset google/fleurs --lang el_gr > results/small_20h_fleurs


# python decode_whisper_cv.py --processor openai/whisper-small  --model openai/whisper-small  --text-key sentence --dataset mozilla-foundation/common_voice_11_0 --lang el > results/small_cv
# python decode_whisper_cv.py --processor openai/whisper-small  --model openai/whisper-small  --text-key transcription --dataset google/fleurs --lang el_gr > results/small_fleurs
# python decode_whisper_cv.py --processor openai/whisper-medium --model openai/whisper-medium --text-key sentence --dataset mozilla-foundation/common_voice_11_0 --lang el > results/medium_cv
# python decode_whisper_cv.py --processor openai/whisper-medium --model openai/whisper-medium --text-key transcription --dataset google/fleurs --lang el_gr > results/medium_fleurs

# python decode_whisper_cv.py --processor openai/whisper-large-v2 --model openai/whisper-large-v2 --text-key sentence --dataset mozilla-foundation/common_voice_11_0 --lang el > results/largev2_cv
# python decode_whisper_cv.py --processor openai/whisper-large-v2 --model openai/whisper-large-v2 --text-key transcription --dataset google/fleurs --lang el_gr > results/largev2_fleurs

# python decode_whisper_cv.py --processor openai/whisper-large-v3 --model openai/whisper-large-v3 --text-key sentence --dataset mozilla-foundation/common_voice_11_0 --lang el > results/largev3_cv
# python decode_whisper_cv.py --processor openai/whisper-large-v3 --model openai/whisper-large-v3 --text-key transcription --dataset google/fleurs --lang el_gr > results/largev3_fleurs
# python decode_whisper_podcast.py --processor whisper-small-el-2h-hf/ --model whisper-small-el-2h-hf/checkpoint-1000 --text-key transcription > results/small_2h_podcast
# python decode_whisper_podcast.py --processor whisper-small-el-5h-hf/ --model whisper-small-el-5h-hf/checkpoint-2375 --text-key transcription > results/small_5h_podcast
# python decode_whisper_podcast.py --processor whisper-small-el-10h-hf/ --model whisper-small-el-10h-hf/checkpoint-5000 --text-key transcription > results/small_10h_podcast
# python decode_whisper_podcast.py --processor whisper-small-el-20h-hf/ --model whisper-small-el-10h-hf/checkpoint-5000 --text-key transcription > results/small_20h_podcast
# python decode_whisper_podcast.py --processor whisper-small-el-all-hf/ --model whisper-small-el-10h-hf/checkpoint-12500 --text-key transcription > results/small_all_podcast

# python decode_whisper_podcast.py --processor whisper-medium-el-2h-hf/ --model whisper-medium-el-2h-hf/checkpoint-1000 --text-key transcription > results/medium_2h_podcast
# python decode_whisper_podcast.py --processor whisper-medium-el-5h-hf/ --model whisper-medium-el-5h-hf/checkpoint-2375 --text-key transcription > results/medium_5h_podcast
# python decode_whisper_podcast.py --processor whisper-medium-el-10h-hf/ --model whisper-medium-el-10h-hf/checkpoint-5000 --text-key transcription > results/medium_10h_podcast
# python decode_whisper_podcast.py --processor whisper-medium-el-20h-hf/ --model whisper-medium-el-20h-hf/checkpoint-10000 --text-key transcription > results/medium_20h_podcast
# # python decode_whisper_podcast.py --processor whisper-medium-el-all-hf/ --model whisper-medium-el-all-hf/checkpoint-12500 --text-key transcription > results/medium_all_podcast
