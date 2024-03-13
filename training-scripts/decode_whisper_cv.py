from transformers import WhisperProcessor, WhisperForConditionalGeneration

from datasets import load_from_disk, load_dataset, Audio
import torch
from evaluate import load
import argparse


# torch.set_num_threads(1)


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process input arguments.")

    # Add arguments with their default values
    parser.add_argument(
        "--processor",
        default="openai/whisper-small",
        help="Processor name (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--model",
        default="openai/whisper-small",
        help="Model name (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--text-key",
        default="transcription",
        help="Key for text input (default: transcription)",
    )
    parser.add_argument(
        "--dataset",
        default="google/fleurs",
        help="Dataset name (default: google/fleurs)",
    )
    parser.add_argument("--lang", default=None, help="Language (default: None)")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


args = parse_args()

processor = WhisperProcessor.from_pretrained(
    args.processor, language="el", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    use_safetensors=True,
    low_cpu_mem_usage=False,
).to("cuda")
model.config.forced_decoder_ids = None


ds = load_dataset(args.dataset, args.lang, split="test", use_auth_token=False)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))


def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(
        [au["array"] for au in audio],
        sampling_rate=audio[0]["sampling_rate"],
        return_tensors="pt",
    ).input_features.to("cuda", dtype=torch.float16)
    batch["reference"] = [
        processor.tokenizer._normalize(t) for t in batch[args.text_key]
    ]
    with torch.no_grad():
        predicted_ids = model.generate(input_features, task="transcribe", language="el")
    transcription = processor.batch_decode(predicted_ids)
    batch["prediction"] = [
        processor.tokenizer._normalize(trans) for trans in transcription
    ]
    return batch


result = ds.map(map_to_pred, batched=True, batch_size=args.batch_size)

for idx, (rf, pr) in enumerate(zip(result["reference"], result["prediction"])):
    print(f"RREF {idx}: " + rf)
    print(f"PRED {idx}: " + pr)

wer = load("wer")
print(
    100 * wer.compute(references=result["reference"], predictions=result["prediction"])
)

female_result = result.filter(lambda b: b["gender"] == "female" or b["gender"] == 1)
male_result = result.filter(lambda b: b["gender"] == "male" or b["gender"] == 0)

print(
    f"female: "
    + str(
        100
        * wer.compute(
            references=female_result["reference"],
            predictions=female_result["prediction"],
        )
    )
)
print(
    f"male: "
    + str(
        100
        * wer.compute(
            references=male_result["reference"], predictions=male_result["prediction"]
        )
    )
)
