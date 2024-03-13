
mkdir -p results_hplg


# CUDA_VISIBLE_DEVICES=0 python decode_whisper_hplg.py --processor whisper-medium-el-5h-hf/  --model whisper-medium-el-5h-hf/checkpoint-2375 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_medium_5h &
# CUDA_VISIBLE_DEVICES=1 python decode_whisper_hplg.py --processor whisper-medium-el-10h-hf/  --model whisper-medium-el-10h-hf/checkpoint-5000 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_medium_10h &
# CUDA_VISIBLE_DEVICES=2 python decode_whisper_hplg.py --processor whisper-medium-el-20h-hf/  --model whisper-medium-el-20h-hf/checkpoint-10000 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_medium_20h &
# CUDA_VISIBLE_DEVICES=3 python decode_whisper_hplg.py --processor whisper-medium-el-all-hf/  --model whisper-medium-el-all-hf/checkpoint-12500 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_medium_50h

# CUDA_VISIBLE_DEVICES=0 python decode_whisper_hplg.py --processor whisper-small-el-5h-hf/  --model whisper-small-el-5h-hf/checkpoint-2375 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_small_5h  &
# CUDA_VISIBLE_DEVICES=1 python decode_whisper_hplg.py --processor whisper-small-el-10h-hf/  --model whisper-small-el-10h-hf/checkpoint-5000 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_small_10h  &
# CUDA_VISIBLE_DEVICES=2 python decode_whisper_hplg.py --processor whisper-small-el-20h-hf/  --model whisper-small-el-20h-hf/checkpoint-5000 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_small_20h  &
# CUDA_VISIBLE_DEVICES=3 python decode_whisper_hplg.py --processor whisper-small-el-all-hf/  --model whisper-small-el-all-hf/checkpoint-12500 --text-key transcription --batch-size 64 --dataset hparl > results_hplg/hparl_small_50h

# CUDA_VISIBLE_DEVICES=0 python decode_whisper_hplg.py --processor whisper-small-el-5h-hf/  --model whisper-small-el-5h-hf/checkpoint-2375 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_small_5h  &
# CUDA_VISIBLE_DEVICES=1 python decode_whisper_hplg.py --processor whisper-small-el-10h-hf/  --model whisper-small-el-10h-hf/checkpoint-5000 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_small_10h  &
CUDA_VISIBLE_DEVICES=2 python decode_whisper_hplg.py --processor whisper-small-el-20h-hf/  --model whisper-small-el-20h-hf/checkpoint-5000 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_small_20h  &
# CUDA_VISIBLE_DEVICES=3 python decode_whisper_hplg.py --processor whisper-small-el-all-hf/  --model whisper-small-el-all-hf/checkpoint-12500 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_small_50h


CUDA_VISIBLE_DEVICES=3 python decode_whisper_hplg.py --processor whisper-medium-el-5h-hf/  --model whisper-medium-el-5h-hf/checkpoint-2375 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_medium_5h  &
# CUDA_VISIBLE_DEVICES=1 python decode_whisper_hplg.py --processor whisper-medium-el-10h-hf/  --model whisper-medium-el-10h-hf/checkpoint-5000 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_medium_10h &
# CUDA_VISIBLE_DEVICES=2 python decode_whisper_hplg.py --processor whisper-medium-el-20h-hf/  --model whisper-medium-el-20h-hf/checkpoint-10000 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_medium_20h  &
# CUDA_VISIBLE_DEVICES=3 python decode_whisper_hplg.py --processor whisper-medium-el-all-hf/  --model whisper-medium-el-all-hf/checkpoint-12500 --text-key transcription --batch-size 64 --dataset logotypografia > results_hplg/logotypografia_medium_50h
#
wait
