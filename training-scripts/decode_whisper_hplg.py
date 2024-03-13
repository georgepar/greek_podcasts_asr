import argparse

import torch
from datasets import Audio, load_dataset, load_from_disk
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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
        help="Dataset path [/path/to/hparl-hf, /path/to/logotypografia-hf]",
    )
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


ds = load_from_disk(args.dataset)


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


# ds = ds.select(list(range(100)))
result = ds.map(map_to_pred, batched=True, batch_size=args.batch_size)
result = result.filter(lambda batch: True if batch["reference"] else False, num_proc=24)

for idx, (rf, pr) in enumerate(zip(result["reference"], result["prediction"])):
    print(f"RREF {idx}: " + rf)
    print(f"PRED {idx}: " + pr)

wer = load("wer")
print(
    "All: "
    + str(
        100
        * wer.compute(references=result["reference"], predictions=result["prediction"])
    )
)
