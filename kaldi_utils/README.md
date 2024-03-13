# Data augmentation with Kaldi

This recipe runs data augmentation with reverbaration or real noises.


## Download noise corpora

Download room impulse responses:

```bash
wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
unzip rir_noises.zip
```

Download musan noise bank:

```bash
wget https://www.openslr.org/resources/17/musan.tar.gz
tar xvf musan.tar.gz
```

**NOTE**: Do not forget to change the `RIR_NOISES` and `MUSAN_PATH` variables in the following scripts:

- `run_augmentation.sh`
- `augment_rvb.sh`
- `augment_noise.sh`
- `augment_music.sh`
- `augment_babble.sh`


## Segment audio (optional)

If your audio is not segmented into utterance level files (i.e. you rely on the segments file and let kaldi do the segmentation while training),
you need to segment your long audio files into utterances.

If you try with the provided example, you can skip this step.

Run:

```bash
./extract_wav_segments_data_dir_eager.sh ${INPUT_KALDI_DIR} ${SEGMENTED_KALDI_DIR} ${SEGMENTED_WAV_FOLDER}
# Example: bash extract_wav_segments_data_dir_eager.sh data/hparl_test data/hparl_segments data/hparl_wavs
```

## Patch the provided example data

To run the scripts for the included example data you first need to patch the provided wav.scp to reflect the path in your system.

Run:

```bash
python kaldipathlib.py change-wav-path --inplace data/cv9_example/wav.scp $(pwd)/data/wavs_cv9
```

This will change the base name for each wav in the wav.scp.

## Run the augmentations

Four scripts are provided to run individual augmentations. Edit the scripts to modify the default configuration.

- `augment_rvb.sh`: Add random reverberation from small or medium rooms.
- `augment_noise.sh`: Add random noise from musan.
- `augment_music.sh`: Add random background music from musan.
- `augment_babble.sh`: Add random background speech from musan.

You can run:

```bash
./augment_noise.sh --kaldi_dir data/cv9_example  # Add background speech in the included example folder
```
The script will create the `data/augmented` folder and the `data/wavs` folder.
Inside the `data/augmented` folder you will find a copy of the original kaldi-style directory `data/augmented/original`
and the noise-augmented directory `data/augmented/noise`. Inside the `data/wavs` folder you will find the augmented wav files.

The rest of the scripts run in the same way.


## Create augmented versions of your dataset

The final step is to create a combined version of your dataset with the augmentations.

Two options are provided:

- mix: For each utterance randomly select one of the augmented wavs or the original wav. This preserves the original number of utterances.

```bash
python kaldipathlib.py mix data/augmented/original data/augmented/babble data/augmented/music data/augmented/noise data/augmented/rvb data/mixed
# Original is mandatory here. If for example you want just the original + noise you can run:
# python kaldipathlib.py mix data/augmented/original data/augmented/noise data/mixed_orig_noise
```

- combine: Concatenate all the augmented versions and the original wavs into a single dataset. This creates a dataset with (N + 1) * #samples, where N the number of augmentations.

```bash
python kaldipathlib.py combine data/augmented/original data/augmented/babble data/augmented/music data/augmented/noise data/augmented/rvb data/combined
```

## I don't care, I want all the augmentation

This script will run all the available augmentations and create a mixed and a combined version of the augmented corpus.

```bash
bash run_augmentation.sh --kaldi_dir data/cv9_example
```
