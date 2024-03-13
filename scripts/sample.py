import argparse
import json
import os
import random
import re
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--hours", type=int)
parser.add_argument("--input-folder", type=str, help="podcast directory")
parser.add_argument("--output-folder", type=str, help="output directory")
parser.add_argument("-debug", default="no")
args = parser.parse_args()
hours = args.hours
debug = args.debug


def convert_to_wav(input_path, output_path):
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-hide_banner",
            "-nostats",
            "-i",
            input_path,
            "-y",
            "-ac",
            "1",
            "-ar",
            "16000",  # Sample rate
            output_path,
        ],
        check=True,
    )


def get_duration_and_meta(directory, input_path):
    meta_file = os.path.splitext(input_path)[0] + ".meta.json"
    meta_path = os.path.join(directory, meta_file)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_sec = result.stdout

    return float(total_sec), meta_path


def get_pod_num(directory):
    count = 0
    podcasts = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # Check if the item is a directory
        if os.path.isdir(item_path):
            count += 1
            podcasts.append(item_path)

    return podcasts, count


def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def sample_videos(directory, output_directory, target_duration_hours):
    podcasts, count = get_pod_num(directory)
    h_per_pod = (target_duration_hours * 3600) / count
    total = 0.0

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for podcast in podcasts:
        podcast_files = []
        selected_podcasts = []
        total_duration = 0.0

        for f in list_full_paths(podcast):
            if f.endswith((".mp3", ".m4a")):
                podcast_files.append(f)

        random.shuffle(podcast_files)
        max_files = len(podcast_files)

        for podcast_file in podcast_files[:max_files]:
            wav_path = os.path.splitext(podcast_file)[0] + ".wav"
            if not os.path.isfile(wav_path):
                convert_to_wav(podcast_file, wav_path)
            podcast_path = os.path.join(directory, wav_path)

            duration, meta_path = get_duration_and_meta(directory, podcast_path)
            total_duration += duration
            if total_duration > h_per_pod:
                total_duration = total_duration - duration
                break

            selected_podcasts.append(podcast_path)
            selected_podcasts.append(meta_path)

        for path in selected_podcasts:
            output_path = os.path.join(output_directory, os.path.basename(path))
            shutil.copy(path, output_path)

        total += total_duration

        print(
            f"Selected {len(selected_podcasts)/2} audio files with a total duration of {total_duration / 3600:.2f} hours from",
            podcast,
        )
    print(f"Sampled a total of {total / 3600:.2f} hours from", directory)


def sample_debug(directory, output_directory):
    podcasts, count = get_pod_num(directory)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    podcast_files = []
    selected_podcasts = []
    for f in list_full_paths(podcasts[0]):
        if f.endswith((".mp3", ".m4a")):
            podcast_files.append(f)

    random.shuffle(podcast_files)
    wav_path = os.path.splitext(podcast_files[0])[0] + ".wav"
    convert_to_wav(podcast_files[0], wav_path)
    podcast_path = os.path.join(directory, wav_path)

    duration, meta_path = get_duration_and_meta(directory, podcast_path)

    selected_podcasts.append(podcast_path)
    selected_podcasts.append(meta_path)
    total_duration = duration
    for path in selected_podcasts:
        output_path = os.path.join(output_directory, os.path.basename(path))
        shutil.copy(path, output_path)

    print(
        f"Selected 1 audio file with a total duration of {total_duration / 3600:.2f} hours from",
        podcasts[0],
    )


if __name__ == "__main__":
    directories = [
        "Arts",
        "Business",
        "Comedy",
        "Education",
        "Government",
        "HealthFitness",
        "History",
        "KidsFamily",
        "Leisure",
        "Music",
        "News",
        "Science",
        "SocietyCulture",
        "Sports",
        "Technology",
        "TrueCrime",
        "TVFilm",
    ]

    for d in directories:
        input_directory = f"{args.input_folder}/{d}"
        output_directory = f"{args.output_folder}/{d}"
        os.makedirs(output_directory, exist_ok=True)
        sample_videos(
            input_directory, output_directory, target_duration_hours=int(hours)
        )
        print("*********************************************************************")
    if debug == "yes":
        for d in directories:
            input_directory = f"{args.input_folder}/{d}"
            output_directory = f"{args.output_folder}/{d}"
            sample_debug(input_directory, output_directory)
            print(
                "*********************************************************************"
            )
