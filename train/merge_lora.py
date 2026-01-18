import argparse
import os

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover - user-facing import error
    raise SystemExit(
        "Missing dependency 'peft'. Install it with: pip install peft"
    ) from exc


def load_tokenizer(adapter_path, base_model):
    for source in (adapter_path, base_model):
        try:
            return AutoTokenizer.from_pretrained(source)
        except Exception:
            continue
    raise SystemExit("Failed to load tokenizer from adapter or base model.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model", default="FacebookAI/xlm-roberta-base", help="Student base model."
    )
    parser.add_argument(
        "--adapter_path",
        default="train/lora_vad_bert/step-150",
        help="LoRA adapter checkpoint path.",
    )
    parser.add_argument(
        "--output_dir",
        default="train/lora_vad_bert_merged_step-150",
        help="Directory to save merged model.",
    )
    parser.add_argument("--num_labels", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.adapter_path):
        raise SystemExit(f"Adapter path not found: {args.adapter_path}")

    config = AutoConfig.from_pretrained(args.base_model)
    if config.num_labels != args.num_labels:
        config.num_labels = args.num_labels

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        config=config,
        ignore_mismatched_sizes=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged = peft_model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir)

    tokenizer = load_tokenizer(args.adapter_path, args.base_model)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
