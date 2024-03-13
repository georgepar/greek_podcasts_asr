import json
import os
import sys
import glob
from collections import Counter, defaultdict
from tqdm import tqdm


def get_most_common_speaker(segment):
    speaker_votes = [word.get("speaker", "SPEAKER") for word in segment["words"]]
    most_common_speaker = Counter(speaker_votes).most_common(1)[0][0]
    return most_common_speaker


def process_json_files(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize dictionaries for content
    wav_scp_dict = {}
    segments_dict = defaultdict(list)
    text_dict = {}
    utt2spk_dict = {}

    for json_file in tqdm(glob.glob(f"{input_folder}/**/*.json", recursive=True)):
        # Extract domain from the subfolder structure
        domain_name = os.path.relpath(os.path.dirname(json_file), input_folder)
        base_name = os.path.basename(json_file).replace(".json", "")
        wav_file_relative_path = os.path.join(domain_name, f"{base_name}.wav")
        wav_file_path = os.path.abspath(
            os.path.join(input_folder, wav_file_relative_path)
        )

        with open(json_file, "r") as f:
            data = json.load(f)
            wav_scp_dict[base_name] = wav_file_path
            seg_idx = 0
            for segment in data["segments"]:
                if "AUTHORWAVE" in segment["text"]:
                    continue
                if float(segment["end"]) - float(segment["start"]) < 0.2:
                    continue
                # Generate utterance ID including the domain

                # Calculate majority speaker ID for the segment including the domain
                try:
                    most_common_speaker = segment["speaker"]
                except KeyError:
                    print("No explicit segment speaker; getting majority vote")
                    most_common_speaker = get_most_common_speaker(segment)
                if most_common_speaker == "SPEAKER":
                    continue

                speaker_id = f"{domain_name}_{base_name}_{most_common_speaker}"
                utt_id = f"{speaker_id}_{seg_idx}"

                # Populate dictionaries
                segments_dict[base_name].append(
                    (utt_id, segment["start"], segment["end"])
                )
                text_dict[utt_id] = " ".join(
                    [word["word"] for word in segment["words"]]
                )
                utt2spk_dict[utt_id] = speaker_id
                seg_idx += 1

    # Write sorted content to files
    with open(os.path.join(output_folder, "wav.scp"), "w") as f:
        for utt_id in sorted(wav_scp_dict):
            f.write(f"{utt_id} {wav_scp_dict[utt_id]}\n")

    with open(os.path.join(output_folder, "segments"), "w") as f:
        for base_name in sorted(segments_dict):
            for segment in sorted(
                segments_dict[base_name], key=lambda x: x[1]
            ):  # Sort by start time
                f.write(f"{segment[0]} {base_name} {segment[1]} {segment[2]}\n")

    with open(os.path.join(output_folder, "text"), "w") as f:
        for utt_id in sorted(text_dict):
            f.write(f"{utt_id} {text_dict[utt_id]}\n")

    with open(os.path.join(output_folder, "utt2spk"), "w") as f:
        for utt_id in sorted(utt2spk_dict):
            f.write(f"{utt_id} {utt2spk_dict[utt_id]}\n")


# Example usage:
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    process_json_files(input_folder, output_folder)
