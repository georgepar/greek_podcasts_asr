
mkdir -p results_hplg
CUDA_VISIBLE_DEVICES=0 python decode_whisper_hplg.py --processor openai/whisper-large-v2 --model openai/whisper-large-v2 --text-key transcription --batch-size 16 --dataset logotypografia > results_hplg/logotypografia_largev2_openai &
CUDA_VISIBLE_DEVICES=1 python decode_whisper_hplg.py --processor openai/whisper-large-v3 --model openai/whisper-large-v3 --text-key transcription --batch-size 16 --dataset logotypografia > results_hplg/logotypografia_largev3_openai  &
CUDA_VISIBLE_DEVICES=2 python decode_whisper_hplg.py --processor openai/whisper-large-v2 --model openai/whisper-large-v2 --text-key transcription --batch-size 16 --dataset hparl > results_hplg/hparl_largev2_openai &
CUDA_VISIBLE_DEVICES=3 python decode_whisper_hplg.py --processor openai/whisper-large-v3 --model openai/whisper-large-v3 --text-key transcription --batch-size 16 --dataset hparl > results_hplg/hparl_largev3_openai &

wait
