import argparse
import csv
import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup


class EmobankDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

def passthrough_collate(batch):
    return batch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_emobank(path, split):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split and row.get("split") != split:
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            labels = [
                float(row["V"]),
                float(row["A"]),
                float(row["D"]),
            ]
            rows.append((text, labels))
    return rows


def collate_batch(batch, tokenizer, device, max_length):
    if isinstance(batch, tuple) and len(batch) == 2:
        texts, labels = batch
    elif isinstance(batch, list):
        texts, labels = zip(*batch)
    else:
        texts = batch[0]
        label_cols = batch[1:]
        labels = list(zip(*label_cols))
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in encoded.items()}
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    return inputs, label_tensor


@torch.no_grad()
def evaluate(model, tokenizer, loader, device, max_length):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    count = 0
    for batch in loader:
        inputs, labels = collate_batch(batch, tokenizer, device, max_length)
        outputs = model(**inputs)
        logits = outputs.logits.float()
        mse = torch.mean((logits - labels) ** 2)
        mae = torch.mean(torch.abs(logits - labels))
        total_mse += mse.item()
        total_mae += mae.item()
        count += 1
    model.train()
    if count == 0:
        return {"mse": None, "mae": None}
    return {"mse": total_mse / count, "mae": total_mae / count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--data_path", default="train/roberta_vad_align/emobank.csv")
    parser.add_argument("--output_dir", default="train/xlm_roberta_emobank")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--eval_every_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--num_labels", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    if config.num_labels != args.num_labels:
        config.num_labels = args.num_labels
    config.problem_type = "regression"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_rows = load_emobank(args.data_path, args.train_split)
    eval_rows = load_emobank(args.data_path, args.eval_split)
    train_loader = DataLoader(
        EmobankDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=passthrough_collate,
    )
    eval_loader = DataLoader(
        EmobankDataset(eval_rows),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=passthrough_collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    scaler = None
    autocast_dtype = None
    if args.dtype == "fp16" and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        autocast_dtype = torch.float16
    elif args.dtype == "bf16" and device.type == "cuda":
        autocast_dtype = torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            inputs, labels = collate_batch(batch, tokenizer, device, args.max_length)
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    outputs = model(**inputs)
                    loss = torch.mean((outputs.logits.float() - labels) ** 2)
            else:
                outputs = model(**inputs)
                loss = torch.mean((outputs.logits.float() - labels) ** 2)

            loss = loss / max(1, args.grad_accum_steps)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                running_loss += loss.item()

                if args.eval_every_steps and global_step % args.eval_every_steps == 0:
                    metrics = evaluate(
                        model, tokenizer, eval_loader, device, args.max_length
                    )
                    if metrics["mse"] is not None:
                        print(
                            f"[step {global_step}] eval_mse={metrics['mse']:.6f} "
                            f"eval_mae={metrics['mae']:.6f}"
                        )

        metrics = evaluate(model, tokenizer, eval_loader, device, args.max_length)
        if metrics["mse"] is not None:
            print(
                f"[epoch {epoch}] eval_mse={metrics['mse']:.6f} "
                f"eval_mae={metrics['mae']:.6f}"
            )

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
