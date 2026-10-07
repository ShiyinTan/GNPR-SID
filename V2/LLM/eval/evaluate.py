"""Generate semantic IDs on a test set and report next-POI metrics.

Example, from the repository root, after LoRA training with template ``qwen``:

    CUDA_VISIBLE_DEVICES=0 python V2/LLM/eval/evaluate.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --adapter ../LLaMA-Factory/saves/nyc-qwen-lora \
        --test_file V1/datasets/nyc/llm_test.json \
        --template qwen \
        --output_dir eval_outputs/nyc-qwen \
        --num_beams 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import DEFAULT_KS, compute_metrics, extract_sid
from prompts import TEMPLATES, build_prompt


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} must be a JSON list of instruction/input/output objects")
    return records


def score_records(
    golds: Sequence[str],
    prediction_lists: Sequence[Sequence[str]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float | int]:
    return compute_metrics(golds, prediction_lists, ks)


def load_prediction_file(path: str) -> tuple[list[str], list[list[str]]]:
    golds: list[str] = []
    prediction_lists: list[list[str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            golds.append(row["gold"])
            predictions = row["predictions"]
            if isinstance(predictions, str):
                predictions = [predictions]
            prediction_lists.append(list(predictions))
    return golds, prediction_lists


def format_metrics(metrics: dict[str, float | int]) -> str:
    lines = [
        f"num_samples: {metrics['num_samples']}",
        f"num_skipped: {metrics['num_skipped']}",
    ]
    for key in ("Acc@1", "Acc@5", "Acc@10", "MRR", "NDCG@5", "NDCG@10"):
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
    return "\n".join(lines)


def truncate_prompt(tokenizer, prompt: str, max_prompt_length: int) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= max_prompt_length:
        return prompt
    return tokenizer.decode(token_ids[-max_prompt_length:], skip_special_tokens=False)


def generate_predictions(
    records: Sequence[dict],
    base_model: str,
    adapter: str | None,
    template: str,
    num_beams: int,
    max_new_tokens: int,
    max_prompt_length: int,
    batch_size: int,
) -> list[list[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    stop_ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|eot_id|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            stop_ids.add(token_id)
    eos_token_id = list(stop_ids)

    prompts = [
        truncate_prompt(
            tokenizer,
            build_prompt(row.get("instruction", ""), row.get("input", ""), template),
            max_prompt_length,
        )
        for row in records
    ]

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    predictions: list[list[str]] = []
    ranges = range(0, len(prompts), batch_size)
    if tqdm is not None:
        ranges = tqdm(ranges, total=(len(prompts) + batch_size - 1) // batch_size, desc="evaluate")

    for start in ranges:
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        for offset in range(len(batch)):
            group = texts[offset * num_beams : (offset + 1) * num_beams]
            predictions.append(group)
    return predictions


def write_outputs(
    output_dir: str,
    records: Sequence[dict],
    prediction_lists: Sequence[Sequence[str]],
    metrics: dict[str, float | int],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    prediction_path = os.path.join(output_dir, "predictions.jsonl")
    with open(prediction_path, "w", encoding="utf-8") as handle:
        for row, predictions in zip(records, prediction_lists):
            gold = row["output"]
            payload = {
                "gold": gold,
                "gold_sid": list(extract_sid(gold) or []),
                "predictions": list(predictions),
                "prediction_sids": [list(extract_sid(text) or []) for text in predictions],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(format_metrics(metrics))
    print(f"predictions: {prediction_path}")
    print(f"metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate next-POI semantic ID predictions")
    parser.add_argument("--test_file", type=str, help="JSON file with instruction/input/output")
    parser.add_argument("--predictions", type=str, help="Existing predictions.jsonl to rescore")
    parser.add_argument("--base_model", type=str, help="Base model name or local path")
    parser.add_argument("--adapter", type=str, default="", help="LoRA adapter directory")
    parser.add_argument("--template", type=str, default="qwen", choices=TEMPLATES)
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0, help="Evaluate only the first N samples")
    args = parser.parse_args()
    if args.predictions:
        return args
    if not args.test_file or not args.base_model:
        parser.error("--test_file and --base_model are required unless --predictions is set")
    if args.num_beams < 1:
        parser.error("--num_beams must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    if args.predictions:
        golds, prediction_lists = load_prediction_file(args.predictions)
        if args.max_samples:
            golds = golds[: args.max_samples]
            prediction_lists = prediction_lists[: args.max_samples]
        metrics = score_records(golds, prediction_lists)
        records = [{"output": gold} for gold in golds]
        write_outputs(args.output_dir, records, prediction_lists, metrics)
        return

    records = load_records(args.test_file)
    if args.max_samples:
        records = records[: args.max_samples]
    prediction_lists = generate_predictions(
        records,
        base_model=args.base_model,
        adapter=args.adapter or None,
        template=args.template,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        batch_size=args.batch_size,
    )
    golds = [row["output"] for row in records]
    metrics = score_records(golds, prediction_lists)
    write_outputs(args.output_dir, records, prediction_lists, metrics)


if __name__ == "__main__":
    main()
