import copy
import os

import torch
from datasets import (Audio, Dataset, DatasetDict, DatasetInfo, Features,
                      GeneratorBasedBuilder, Split, SplitGenerator, Value,
                      Version)

# torch.set_num_threads(1)


def parse_kaldi_file(kaldi_file, excluded_ids=[]):
    excluded_ids_ = copy.deepcopy(excluded_ids)
    with open(kaldi_file) as f:
        data = []
        for s in f:
            dat = s.strip().split(maxsplit=1)
            if dat[0] in excluded_ids_:
                continue
            if len(dat) != 2:
                print(dat)
                excluded_ids_.append(dat[0])
                continue
            data.append(dat)
    return dict(data), excluded_ids_


class DatasetWithAudio(GeneratorBasedBuilder):
    VERSION = Version("1.0.0")

    def __init__(self, data_path, **kwargs):
        """
        Dataset builder that accepts custom file paths.

        :param data_dir: Base directory containing the dataset splits (train_kaldi, dev_kaldi, test_kaldi).
        """
        super().__init__(**kwargs)  # Initialize the parent class
        self.train_data_dir = f"{data_path}/train_kaldi_segmented"
        self.train2h_data_dir = f"{data_path}/gpc2_train_kaldi_segmented"
        self.train5h_data_dir = f"{data_path}/gpc5_train_kaldi_segmented"
        self.train10h_data_dir = f"{data_path}/gpc10_train_kaldi_segmented"
        self.train20h_data_dir = f"{data_path}/gpc20_train_kaldi_segmented"
        self.dev_data_dir = f"{data_path}/dev_kaldi_segmented"
        self.test_data_dir = f"{data_path}/test_kaldi_segmented"

    def _info(self):
        return DatasetInfo(
            description="This dataset includes audio recordings, their transcriptions, speaker IDs, and domain information.",
            features=Features(
                {
                    "audio": Audio(
                        sampling_rate=16_000
                    ),  # Adjust the sampling rate according to your audio files
                    "transcription": Value("string"),  # Transcription of the audio
                    "id": Value("string"),  # Unique identifier for each sample
                    "speaker": Value("string"),  # Speaker ID
                    "domain": Value("string"),  # Domain or category of the audio
                }
            ),
            supervised_keys=(
                "audio",
                "transcription",
            ),  # Indicates which features are input and target for supervised learning
            version=self.VERSION,  # Optional: Version of the dataset
        )

    def _split_generators(self, _):
        """
        Returns SplitGenerators for each dataset split, using dynamically specified paths.
        """
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"split_dir": self.train_data_dir},
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={"split_dir": self.dev_data_dir},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"split_dir": self.test_data_dir},
            ),
            SplitGenerator(
                name="gpc2_train",
                gen_kwargs={"split_dir": self.train2h_data_dir},
            ),
            SplitGenerator(
                name="gpc5_train",
                gen_kwargs={"split_dir": self.train5h_data_dir},
            ),
            SplitGenerator(
                name="gpc10_train",
                gen_kwargs={"split_dir": self.train10h_data_dir},
            ),
            SplitGenerator(
                name="gpc20_train",
                gen_kwargs={"split_dir": self.train20h_data_dir},
            ),
        ]

    def _generate_examples(self, split_dir):
        """
        Yields examples from a given dataset split.

        :param split_dir: Directory for the current split.
        """
        # Define paths based on split_dir
        texts_file = os.path.join(split_dir, "text")
        utt2spk_file = os.path.join(split_dir, "utt2spk")
        wavs_scp_file = os.path.join(split_dir, "wav.scp")

        # Load texts and utt2spk mappings
        # excluded_
        texts, excluded_ids = parse_kaldi_file(texts_file, excluded_ids=[])
        utt2spk, excluded_ids = parse_kaldi_file(
            utt2spk_file, excluded_ids=excluded_ids
        )
        wavs, _ = parse_kaldi_file(wavs_scp_file, excluded_ids=excluded_ids)

        for key, uttid in enumerate(texts.keys()):
            transcript = texts[uttid]
            speaker = utt2spk[uttid]
            domain = uttid.split("_")[0]
            audio_path = wavs[uttid]  # Adjust based on how _load_scp returns paths

            yield key, {
                "audio": {"path": audio_path},
                "transcription": transcript,
                "id": uttid,
                "speaker": speaker,
                "domain": domain,
            }


if __name__ == "__main__":
    dataset_builder = DatasetWithAudio(
        data_path="./gpc-50-all/",
        dataset_name="greek_podcast_dataset",
        # writer_batch_size=1024,
    )
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    dataset.save_to_disk("./greek_podcast_dataset", num_proc=24)
