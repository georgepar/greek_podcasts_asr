import argparse
import glob
import os
import random
import shutil  # Import shutil to copy files
from pydub import AudioSegment
from joblib import Parallel, delayed
from tqdm import tqdm


def process_subfolder(input_subfolder, output_subfolder, target_duration_hours):
    files = glob.glob(os.path.join(input_subfolder, "**/*.wav"), recursive=True)
    random.shuffle(files)

    total_duration_sec = target_duration_hours * 3600
    current_duration_sec = 0

    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    for file in tqdm(files):
        # Calculate the duration without altering the file
        sound = AudioSegment.from_file(file)
        file_duration_sec = len(sound) / 1000

        # Update current duration
        current_duration_sec += file_duration_sec

        if current_duration_sec >= total_duration_sec:
            break

        # Directly copy the WAV file
        shutil.copy(file, os.path.join(output_subfolder, os.path.basename(file)))

        # Check and copy the corresponding JSON file if it exists
        json_file = file.replace(".wav", ".json")
        if os.path.exists(json_file):
            shutil.copy(json_file, os.path.join(output_subfolder, os.path.basename(json_file)))


def main(input_folder, output_folder, hours):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    Parallel(n_jobs=-1)(
        delayed(process_subfolder)(
            subfolder, os.path.join(output_folder, os.path.basename(subfolder)), hours
        )
        for subfolder in subfolders
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy WAV and JSON files.")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder path")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder path")
    parser.add_argument("--hours", type=float, required=True, help="Total hours of audio to sample")

    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.hours)

