import argparse
import glob
import os
import random

from joblib import Parallel, delayed
from pydub import AudioSegment
from tqdm import tqdm


def process_subfolder(input_subfolder, output_subfolder, target_duration_hours):
    # files = [
    #     os.path.join(input_subfolder, f)
    #     for f in os.listdir(input_subfolder)
    #     if f.endswith(".wav")
    # ]
    files = glob.glob(os.path.join(input_subfolder, "**/*.wav"))
    random.shuffle(files)

    total_duration_sec = target_duration_hours * 3600
    current_duration_sec = 0

    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    for file in tqdm(files):
        if current_duration_sec >= total_duration_sec:
            break

        sound = AudioSegment.from_file(file)
        current_duration_sec += len(sound) / 1000

        if current_duration_sec > total_duration_sec:
            # Trim the audio if it exceeds the target duration
            sound = sound[
                : int(
                    (total_duration_sec - (current_duration_sec - len(sound) / 1000))
                    * 1000
                )
            ]

        # Convert to 16000 Hz, single-channel (mono)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(
            os.path.join(output_subfolder, os.path.basename(file)), format="wav"
        )


def main(input_folder, output_folder, hours):
    # Create the output folder structure
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    # for subfolder in subfolders:
    #     process_subfolder(
    #         subfolder, os.path.join(output_folder, os.path.basename(subfolder)), hours
    #     )
    # Parallel processing of subfolders
    Parallel(n_jobs=-1)(
        delayed(process_subfolder)(
            subfolder, os.path.join(output_folder, os.path.basename(subfolder)), hours
        )
        for subfolder in subfolders
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument("--input_folder", type=str, help="Input folder path")
    parser.add_argument("--output_folder", type=str, help="Output folder path")
    parser.add_argument("--hours", type=float, help="Total hours of audio to sample")

    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.hours)
