import argparse
import io
import zipfile

from transformers import AutoTokenizer


def update_stats(stats, input_ids, unk_id):
    stats["total_tokens"] += len(input_ids)
    stats["unk_tokens"] += sum(1 for tok in input_ids if tok == unk_id)
    stats["total_lines"] += 1
    if unk_id in input_ids:
        stats["lines_with_unk"] += 1


def process_batch(texts, tokenizer, unk_id, stats):
    if not texts:
        return
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    for input_ids in encoded["input_ids"]:
        update_stats(stats, input_ids, unk_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--data_zip", default="train/en-zh_cn.txt.zip")
    parser.add_argument("--en_file", default="OpenSubtitles.en-zh_cn.en")
    parser.add_argument("--zh_file", default="OpenSubtitles.en-zh_cn.zh_cn")
    parser.add_argument("--max_lines", type=int, default=20000)
    parser.add_argument("--min_chars", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        raise SystemExit("Tokenizer has no unk_token_id; cannot compute UNK ratio.")

    stats_en = {"total_tokens": 0, "unk_tokens": 0, "total_lines": 0, "lines_with_unk": 0}
    stats_zh = {"total_tokens": 0, "unk_tokens": 0, "total_lines": 0, "lines_with_unk": 0}
    batch_en = []
    batch_zh = []

    with zipfile.ZipFile(args.data_zip) as zf:
        with zf.open(args.en_file) as en_raw, zf.open(args.zh_file) as zh_raw:
            en_f = io.TextIOWrapper(en_raw, encoding=args.encoding, errors="ignore")
            zh_f = io.TextIOWrapper(zh_raw, encoding=args.encoding, errors="ignore")
            for idx, (en_line, zh_line) in enumerate(zip(en_f, zh_f)):
                if args.max_lines and idx >= args.max_lines:
                    break
                en = en_line.rstrip("\n")
                zh = zh_line.rstrip("\n")
                if len(en) < args.min_chars or len(zh) < args.min_chars:
                    continue
                batch_en.append(en)
                batch_zh.append(zh)
                if len(batch_en) >= args.batch_size:
                    process_batch(batch_en, tokenizer, unk_id, stats_en)
                    process_batch(batch_zh, tokenizer, unk_id, stats_zh)
                    batch_en = []
                    batch_zh = []

    process_batch(batch_en, tokenizer, unk_id, stats_en)
    process_batch(batch_zh, tokenizer, unk_id, stats_zh)

    def report(label, stats):
        total_tokens = stats["total_tokens"]
        unk_tokens = stats["unk_tokens"]
        total_lines = stats["total_lines"]
        lines_with_unk = stats["lines_with_unk"]
        token_ratio = (unk_tokens / total_tokens) if total_tokens else 0.0
        line_ratio = (lines_with_unk / total_lines) if total_lines else 0.0
        print(f"{label}:")
        print(f"  lines={total_lines} tokens={total_tokens}")
        print(f"  unk_tokens={unk_tokens} unk_ratio={token_ratio:.4f}")
        print(f"  lines_with_unk={lines_with_unk} line_ratio={line_ratio:.4f}")

    report("EN", stats_en)
    report("ZH", stats_zh)


if __name__ == "__main__":
    main()
