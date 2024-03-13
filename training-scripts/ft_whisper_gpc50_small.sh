DATASET_PATH=$1

export CUDA_VISIBLE_DEVICES=0
python run_speech_recognition_seq2seq.py \
	--model_name_or_path="openai/whisper-small" \
	--dataset_name=$DATASET_PATH \
	--dataset_config_name="el" \
	--language="el" \
	--train_split_name="gpc50_train" \
	--max_steps="25000" \
	--output_dir="./whisper-small-el-gpc50-hf" \
	--per_device_train_batch_size="16" \
	--gradient_accumulation_steps="4" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--save_strategy="steps" \
	--save_steps="100" \
    --save_total_limit 10 \
	--generation_max_length="225" \
	--preprocessing_num_workers="16" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="text" \
	--freeze_feature_encoder="False" \
	--freeze_encoder="True" \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--predict_with_generate \
    --dataloader_num_workers=8
