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
        help="Path to test podcast dataset",
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


ds = load_from_disk(args.dataset)

DOMAINS = {
    "Arts",
    "Business",
    "Comedy",
    "Education",
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
}


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
    # for domain in list(DOMAINS):
    #     rkey = f"{domain}_reference"
    #     pkey = f"{domain}_prediction"
    #     batch[rkey] = [""] * args.batch_size
    #     batch[pkey] = [""] * args.batch_size
    # for idx, (domain, ref, pred) in enumerate(
    #     zip(batch["domain"], batch["reference"], batch["prediction"])
    # ):
    #     rkey = f"{domain}_reference"
    #     pkey = f"{domain}_prediction"
    #     batch[rkey][idx] = ref
    #     batch[pkey][idx] = pred
    return batch


result = ds.map(
    map_to_pred, batched=True, batch_size=args.batch_size
)

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


for domain in list(DOMAINS):
    domain_result = result.filter(lambda b: b["domain"] == domain)
    if len(domain_result) == 0:
        continue
    print(
        f"{domain}: "
        + str(
            100
            * wer.compute(
                references=domain_result["reference"],
                predictions=domain_result["prediction"],
            )
        )
    )
