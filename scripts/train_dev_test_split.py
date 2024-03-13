import argparse
import glob
import os
import random
import shutil
import hashlib

from joblib import Parallel, delayed
from pydub import AudioSegment
from tqdm import tqdm


def generate_sha1(filename):
    sha1 = hashlib.sha1()
    sha1.update(filename.encode("utf-8"))
    return sha1.hexdigest()


def process_subfolder(
    input_subfolder,
    train_output_subfolder,
    test_output_subfolder,
    dev_output_subfolder,
    test_duration_hours,
    dev_duration_hours,
    rename_sha=False,
    shuffle_files=True,
):
    files = glob.glob(os.path.join(input_subfolder, "*.wav"))
    if shuffle_files:
        random.shuffle(files)

    test_duration_sec = test_duration_hours * 3600
    dev_duration_sec = dev_duration_hours * 3600
    current_test_duration_sec = 0
    current_dev_duration_sec = 0

    for file in tqdm(files):
        sound = AudioSegment.from_file(file)
        file_duration_sec = len(sound) / 1000
        original_filename = os.path.basename(file)
        if rename_sha:
            hash_name = generate_sha1(original_filename) + '.wav'
        else:
            hash_name = original_filename

        # Determine if the file should go into the test, dev, or train set
        if current_test_duration_sec + file_duration_sec <= test_duration_sec:
            target_folder = test_output_subfolder
            current_test_duration_sec += file_duration_sec
        elif current_dev_duration_sec + file_duration_sec <= dev_duration_sec:
            target_folder = dev_output_subfolder
            current_dev_duration_sec += file_duration_sec
        else:
            target_folder = train_output_subfolder

        # Ensure the target folder exists
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Rename and copy .wav file
        new_file_path = os.path.join(target_folder, hash_name)
        shutil.copy(file, new_file_path)

        # Rename and copy corresponding .json file, if it exists
        json_file = file.replace(".wav", ".json")
        if os.path.exists(json_file):
            new_json_path = os.path.join(target_folder, hash_name.replace('.wav', '.json'))
            shutil.copy(json_file, new_json_path)


def main(input_folder, output_folder, test_duration_hours, dev_duration_hours, rename_sha, shuffle):
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    Parallel(n_jobs=-1)(
        delayed(process_subfolder)(
            subfolder,
            os.path.join(output_folder, "train", os.path.basename(subfolder)),
            os.path.join(output_folder, "test", os.path.basename(subfolder)),
            os.path.join(output_folder, "dev", os.path.basename(subfolder)),
            test_duration_hours,
            dev_duration_hours,
            rename_sha,
            shuffle,
        )
        for subfolder in subfolders
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process audio files for training, test, and dev split, with options for renaming and shuffling."
    )
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder path")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder path")
    parser.add_argument("--test_hours", type=float, required=True, help="Total hours of audio for the test set per domain")
    parser.add_argument("--dev_hours", type=float, required=True, help="Total hours of audio for the dev set per domain")
    parser.add_argument("--rename_sha", action="store_true", help="Rename files based on SHA-1 hash")
    parser.add_argument("--shuffle", action="store_false", help="Shuffle files before splitting")

    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.test_hours, args.dev_hours, args.rename_sha, args.shuffle)

