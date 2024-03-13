# The Greek podcast corpus: Competitive speech models for low-resourced languages with weakly supervised data

Code for the paper "The Greek podcast corpus: Competitive speech models
for  
low-resourced languages with weakly supervised data". Submitted  
Interspeech, 2024.

## File structure

    .  
    ├── download_asr.sh                  # Download audio from rss feeds  
    ├── download_individual_rss_list.sh  
    ├── download_tts.sh  
    ├── README.md                        # This file  
    ├── kaldi_utils                      # Utilities for audio segmentation  
    ├── rss-lists                        # Include downloaded RSS feeds for reproducibility  
    │   ├── asr  
    │   │   ├── Arts.txt  
    │   │   ├── Business.txt  
    │   │   ├── Comedy.txt  
    │   │   ├── Education.txt  
    │   │   ├── Government.txt  
    │   │   ├── HealthFitness.txt  
    │   │   ├── History.txt  
    │   │   ├── KidsFamily.txt  
    │   │   ├── Leisure.txt  
    │   │   ├── Music.txt  
    │   │   ├── News.txt  
    │   │   ├── Science.txt  
    │   │   ├── SocietyCulture.txt  
    │   │   ├── Sports.txt  
    │   │   ├── Technology.txt  
    │   │   ├── TrueCrime.txt  
    │   │   └── TVFilm.txt  
    │   └── tts  
    │       ├── audiobooks.txt  
    │       └── political.txt  
    ├── scrape_rss                       # Scraper to download new RSS feeds  
    └── scripts                          # Scripts for data creation and preprocessing  
        ├── create_subset.py  
        ├── get_subset.py  
        ├── hf_data_gen.py  
        ├── sample.py  
        ├── to_kaldi.py  
        ├── train_dev_test_split.py  
        └── transcribe.sh  

## Collect RSS feeds

In the `rss-lists` folder we include the RSS feeds we collected per task
(`asr`  
and `tts`). In the `asr` folder the feeds are split per domain.

We also include the scrapy crawler so that you can collect more RSS
feeds in the  
folder `scrape_rss`.

Run:

``` bash
cd scrape_rss  
scrapy crawl parss -o output.json -a lang=el  
```

## Download audio from RSS feeds

Run

``` bash
download_asr.sh  
```

## Scripts for data preparation

-   Step 1: Get a random subset (50 hours) per domain

``` bash
mkdir -p gpc-50; python scripts/sample.py --input_folder $(pwd)/gpc --output_folder $(pwd)/gpc-50 --hours 50  
```

-   Step 2: Transcribe the podcasts

``` bash
bash scripts/transcribe.sh gpc-50  
```

-   Step 3: Create train-validation-test split

``` bash
python scripts/train_dev_test_split.py --input_folder gpc-50 --output_folder gpc-50-all --dev_hours 0.3 --test_hours 1 --rename_sha --shuffle  
```

-   Step 4: Create subsets

``` bash
mkdir gpc-50-all/gpc-20-train; python scripts/get_subset.py --input_folder gpc-50-all/train --output_folder gpc-50-all/gpc-20-train/ --hours 20  
mkdir gpc-50-all/gpc-10-train; python scripts/get_subset.py --input_folder gpc-50-all/train --output_folder gpc-50-all/gpc-10-train/ --hours 10  
mkdir gpc-50-all/gpc-5-train; python scripts/get_subset.py --input_folder gpc-50-all/train --output_folder gpc-50-all/gpc-5-train/ --hours 5  
mkdir gpc-50-all/gpc-2-train; python scripts/get_subset.py --input_folder gpc-50-all/train --output_folder gpc-50-all/gpc-2-train/ --hours 2  
```

-   Step 5: Convert to kaldi format

``` bash
python scripts/to_kaldi.py gpc-50-all/train gpc-50-all/train_kaldi  
python scripts/to_kaldi.py gpc-50-all/test gpc-50-all/test_kaldi  
python scripts/to_kaldi.py gpc-50-all/dev gpc-50-all/dev_kaldi  
python scripts/to_kaldi.py gpc-50-all/gpc-20-train gpc-50-all/gpc20_train_kaldi  
python scripts/to_kaldi.py gpc-50-all/gpc-10-train gpc-50-all/gpc10_train_kaldi  
python scripts/to_kaldi.py gpc-50-all/gpc-5-train gpc-50-all/gpc5_train_kaldi  
python scripts/to_kaldi.py gpc-50-all/gpc-2-train gpc-50-all/gpc2_train_kaldi  
```

-   Step 6: Extract audio segments (Must have a valid Kaldi installation
    -> export  
    KALDI_PATH=/path/to/kaldi)

``` bash
cd kaldi_utils  
bash extract_wav_segments_data_dir_eager.sh ../gpc-50-all/train_kaldi ../gpc-50-all/train_kaldi_segmented ../gpc-50-all/train_segmented  
bash extract_wav_segments_data_dir_eager.sh ../gpc-50-all/test_kaldi ../gpc-50-all/test_kaldi_segmented ../gpc-50-all/test_segmented  
bash extract_wav_segments_data_dir_eager.sh ../gpc-50-all/dev_kaldi ../gpc-50-all/dev_kaldi_segmented ../gpc-50-all/dev_segmented  
bash extract_wav_segments_data_dir_eager.sh ../gpc-50-all/gpc20_train_kaldi ../gpc-50-all/gpc20_train_kaldi_segmented ../gpc-50-all/gpc20_train_segmented  
bash extract_wav_segments_data_dir_eager.sh ../gpc-50-all/gpc10_train_kaldi ../gpc-50-all/gpc10_train_kaldi_segmented ../gpc-50-all/gpc10_train_segmented  
bash extract_wav_segments_data_dir_eager.sh ../gpc-50-all/gpc5_train_kaldi ../gpc-50-all/gpc5_train_kaldi_segmented ../gpc-50-all/gpc5_train_segmented  
bash extract_wav_segments_data_dir_eager.sh ../gpc-50-all/gpc2_train_kaldi ../gpc-50-all/gpc2_train_kaldi_segmented ../gpc-50-all/gpc2_train_segmented  
```

-   Step 5: Convert to huggingface format

``` bash
python scripts/hf_data_gen.py  
```

## Training the whisper models

Select the model, the subset and set the dataset path, and then run:

``` bash
export MODEL=small # or medium  
export TRAINING_SUBSET=gpc50  # or gpc2, gpc5, gpc10, gpc20  
export DATASET_PATH=$(pwd)/greek_podcast_dataset  
  
cd training-scripts  
bash ft_whisper_${TRAINING_SUBSET}_${MODEL}.sh  
```

## Evaluating the models on the test sets

-   For common voice and fleurs

``` bash
export CHECKPOINT_STEPS=3000 # The latest checkpoint  
cd training-scripts  
python decode_whisper_cv.py --processor ./whisper-${MODEL}-el-${TRAINING_SUBSET}-hf --model ./whisper-${MODEL}-el-${TRAINING_SUBSET}-hf/checkpoint-${CHECKPOINT_STEPS} --text-key sentence --dataset mozilla-foundation/common_voice_11_0 --lang el  
python decode_whisper_cv.py --processor ./whisper-${MODEL}-el-${TRAINING_SUBSET}-hf --model ./whisper-${MODEL}-el-${TRAINING_SUBSET}-hf/checkpoint-${CHECKPOINT_STEPS} --text-key transcription --dataset google/fleurs --lang el  
```

-   For hparl and logotypografia (assuming you have downloaded and
    converted the  
    datasets to huggingface format in ./hparl-test-hf and  
    ./logotypografia-test-hf)

``` bash
export CHECKPOINT_STEPS=3000 # The latest checkpoint  
cd training-scripts  
python decode_whisper_hplg.py --processor ./whisper-${MODEL}-el-${TRAINING_SUBSET}-hf --model ./whisper-${MODEL}-el-${TRAINING_SUBSET}-hf/checkpoint-${CHECKPOINT_STEPS} --text-key transcription --dataset ./hparl-test-hf --lang el  
python decode_whisper_hplg.py --processor ./whisper-${MODEL}-el-${TRAINING_SUBSET}-hf --model ./whisper-${MODEL}-el-${TRAINING_SUBSET}-hf/checkpoint-${CHECKPOINT_STEPS} --text-key transcription --dataset ./logotypografia-test-hf --lang el  
```

-   For the greek podcast dataset

``` bash
export CHECKPOINT_STEPS=3000 # The latest checkpoint  
cd training-scripts  
python decode_whisper_podcast.py --processor ./whisper-${MODEL}-el-${TRAINING_SUBSET}-hf --model ./whisper-${MODEL}-el-${TRAINING_SUBSET}-hf/checkpoint-${CHECKPOINT_STEPS} --text-key transcription --dataset ../greek_podcast_dataset/test --lang el  
```
