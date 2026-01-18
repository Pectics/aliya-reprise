import argparse
import io
import json
import os
import random
import zipfile
from typing import List

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError as exc:  # pragma: no cover - import error is user-facing
    raise SystemExit(
        "Missing dependency 'peft'. Install it with: pip install peft"
    ) from exc


class ParallelZipDataset(IterableDataset):
    def __init__(
        self,
        zip_path,
        en_file,
        zh_file,
        max_lines=None,
        min_chars=1,
        shuffle_buffer=0,
        encoding="utf-8",
    ):
        self.zip_path = zip_path
        self.en_file = en_file
        self.zh_file = zh_file
        self.max_lines = max_lines
        self.min_chars = min_chars
        self.shuffle_buffer = shuffle_buffer
        self.encoding = encoding

    def _iter_pairs(self):
        with zipfile.ZipFile(self.zip_path) as zf:
            with zf.open(self.en_file) as en_raw, zf.open(self.zh_file) as zh_raw:
                en_f = io.TextIOWrapper(en_raw, encoding=self.encoding, errors="ignore")
                zh_f = io.TextIOWrapper(zh_raw, encoding=self.encoding, errors="ignore")
                for idx, (en_line, zh_line) in enumerate(zip(en_f, zh_f)):
                    if self.max_lines is not None and idx >= self.max_lines:
                        break
                    en = en_line.rstrip("\n")
                    zh = zh_line.rstrip("\n")
                    if self.min_chars and (len(en) < self.min_chars or len(zh) < self.min_chars):
                        continue
                    yield en, zh

    def __iter__(self):
        if self.shuffle_buffer and self.shuffle_buffer > 1:
            buffer = []
            for item in self._iter_pairs():
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)
                    while buffer:
                        yield buffer.pop()
            if buffer:
                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        else:
            yield from self._iter_pairs()


def collate_pairs(batch):
    ens, zhs = zip(*batch)
    return list(ens), list(zhs)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(texts, tokenizer, device, max_length):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoded.items()}


def infer_target_modules(model) -> List[str]:
    module_names = set(name.split(".")[-1] for name, _ in model.named_modules())
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    candidate_groups = []
    if model_type in {"xlm-roberta", "roberta"}:
        candidate_groups.extend(
            [
                ["query", "key", "value"],
                ["q_proj", "k_proj", "v_proj", "out_proj"],
            ]
        )
    elif model_type in {"bert"}:
        candidate_groups.extend([["query", "key", "value"]])
    candidate_groups.extend(
        [
            ["query", "key", "value"],
            ["q_proj", "k_proj", "v_proj", "out_proj"],
        ]
    )
    for group in candidate_groups:
        found = [name for name in group if name in module_names]
        if found:
            return found
    return []


def resolve_target_modules(raw_modules: List[str], model) -> List[str]:
    module_names = set(name.split(".")[-1] for name, _ in model.named_modules())
    if raw_modules:
        requested: List[str] = [name.strip() for name in raw_modules if name.strip()]
        found = [name for name in requested if name in module_names]
        if found:
            return found
        synonym_map = {
            "q_proj": "query",
            "k_proj": "key",
            "v_proj": "value",
            "out_proj": "dense",
        }
        mapped = [synonym_map.get(name, name) for name in requested]
        found = [name for name in mapped if name in module_names]
        if found:
            return list(dict.fromkeys(found))
        expected = [
            name
            for name in (
                "query",
                "key",
                "value",
                "dense",
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
            )
            if name in module_names
        ]
        expected_hint = ",".join(expected) if expected else "query,key,value"
        raise SystemExit(
            "No LoRA target modules matched. Try --target_modules "
            f"{expected_hint}"
        )
    return infer_target_modules(model)


def find_modules_to_save(model):
    modules = []
    for name in ["classifier", "score"]:
        if hasattr(model, name):
            modules.append(name)
    return modules


def compute_mse(a, b):
    return torch.mean((a - b) ** 2)


@torch.no_grad()
def teacher_forward(teacher, inputs):
    outputs = teacher(**inputs)
    return outputs.logits.float()


@torch.no_grad()
def evaluate(
    student, teacher, student_tokenizer, teacher_tokenizer, dataset, device, max_length, max_batches
):
    student.eval()
    teacher.eval()
    total_old = 0.0
    total_zh = 0.0
    count = 0
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_pairs)
    for batch_idx, (ens, zhs) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        en_inputs_student = tokenize(ens, student_tokenizer, device, max_length)
        zh_inputs_student = tokenize(zhs, student_tokenizer, device, max_length)
        en_inputs_teacher = tokenize(ens, teacher_tokenizer, device, max_length)
        teacher_en = teacher_forward(teacher, en_inputs_teacher)
        student_en = student(**en_inputs_student).logits.float()
        student_zh = student(**zh_inputs_student).logits.float()
        total_old += torch.mean(torch.norm(student_en - teacher_en, dim=-1)).item()
        total_zh += torch.mean(torch.norm(student_zh - teacher_en, dim=-1)).item()
        count += 1
    student.train()
    if count == 0:
        return {"E_old": None, "E_zh": None}
    return {"E_old": total_old / count, "E_zh": total_zh / count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="RobroKools/vad-bert")
    parser.add_argument("--teacher_model", default="")
    parser.add_argument("--student_model", default="")
    parser.add_argument("--teacher_tokenizer", default="")
    parser.add_argument("--student_tokenizer", default="")
    parser.add_argument("--data_zip", default="train/en-zh_cn.txt.zip")
    parser.add_argument("--en_file", default="OpenSubtitles.en-zh_cn.en")
    parser.add_argument("--zh_file", default="OpenSubtitles.en-zh_cn.zh_cn")
    parser.add_argument("--output_dir", default="train/lora_vad_bert")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_lines", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--min_chars", type=int, default=2)
    parser.add_argument("--parallel_ratio", type=float, default=0.5)
    parser.add_argument("--lambda_zh", type=float, default=1.0)
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--lambda_retain", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="")
    parser.add_argument("--train_classifier", action="store_true")
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--shuffle_buffer", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model_name = args.teacher_model or args.model_name
    student_model_name = args.student_model or args.model_name
    teacher_tokenizer_name = args.teacher_tokenizer or teacher_model_name
    student_tokenizer_name = args.student_tokenizer or student_model_name

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer_name)
    student_tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_name)

    student_config = AutoConfig.from_pretrained(student_model_name)
    if student_config.num_labels != args.num_labels:
        student_config.num_labels = args.num_labels
    base_model = AutoModelForSequenceClassification.from_pretrained(
        student_model_name,
        config=student_config,
        ignore_mismatched_sizes=True,
    )
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise SystemExit(f"Resume path not found: {args.resume_from}")
        student = PeftModel.from_pretrained(
            base_model, args.resume_from, is_trainable=True
        )
        if args.train_classifier:
            for name in ["classifier", "score"]:
                module = getattr(student, name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
    else:
        target_modules = resolve_target_modules(
            [name.strip() for name in args.target_modules.split(",") if name.strip()],
            base_model,
        )
        if not target_modules:
            raise SystemExit(
                "Could not infer LoRA target modules. Provide --target_modules."
            )

        modules_to_save = (
            find_modules_to_save(base_model) if args.train_classifier else []
        )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            bias="none",
            task_type="SEQ_CLS",
        )
        student = get_peft_model(base_model, lora_config)
    student.to(device)

    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_model_name)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    dataset = ParallelZipDataset(
        args.data_zip,
        args.en_file,
        args.zh_file,
        max_lines=args.max_lines,
        min_chars=args.min_chars,
        shuffle_buffer=args.shuffle_buffer,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_pairs,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    total_optim_steps = max(1, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_optim_steps
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

    student.train()
    global_step = 0
    accum_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            if args.max_steps and global_step >= args.max_steps:
                break

            ens, zhs = batch
            batch_len = len(ens)
            n_parallel = int(round(batch_len * args.parallel_ratio))
            n_parallel = max(0, min(batch_len, n_parallel))
            n_replay = batch_len - n_parallel

            parallel_en = ens[:n_parallel]
            parallel_zh = zhs[:n_parallel]
            replay_en = ens[n_parallel:]

            loss_zh = torch.tensor(0.0, device=device)
            loss_align = torch.tensor(0.0, device=device)
            loss_retain = torch.tensor(0.0, device=device)

            if n_parallel > 0:
                en_inputs_student = tokenize(
                    parallel_en, student_tokenizer, device, args.max_length
                )
                zh_inputs_student = tokenize(
                    parallel_zh, student_tokenizer, device, args.max_length
                )
                en_inputs_teacher = tokenize(
                    parallel_en, teacher_tokenizer, device, args.max_length
                )
                with torch.no_grad():
                    teacher_en = teacher_forward(teacher, en_inputs_teacher)
                if autocast_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        student_en = student(**en_inputs_student).logits.float()
                        student_zh = student(**zh_inputs_student).logits.float()
                else:
                    student_en = student(**en_inputs_student).logits.float()
                    student_zh = student(**zh_inputs_student).logits.float()
                loss_zh = compute_mse(student_zh, teacher_en)
                loss_align = compute_mse(student_en, student_zh)

            if n_replay > 0:
                replay_inputs_student = tokenize(
                    replay_en, student_tokenizer, device, args.max_length
                )
                replay_inputs_teacher = tokenize(
                    replay_en, teacher_tokenizer, device, args.max_length
                )
                with torch.no_grad():
                    teacher_replay = teacher_forward(teacher, replay_inputs_teacher)
                if autocast_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        student_replay = student(**replay_inputs_student).logits.float()
                else:
                    student_replay = student(**replay_inputs_student).logits.float()
                loss_retain = compute_mse(student_replay, teacher_replay)

            total_loss = (
                args.lambda_zh * loss_zh
                + args.lambda_align * loss_align
                + args.lambda_retain * loss_retain
            )
            total_loss = total_loss / max(1, args.grad_accum_steps)

            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            accum_step += 1
            if accum_step % args.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.eval_every and global_step % args.eval_every == 0:
                    eval_dataset = ParallelZipDataset(
                        args.data_zip,
                        args.en_file,
                        args.zh_file,
                        max_lines=args.eval_batches,
                        min_chars=args.min_chars,
                    )
                    metrics = evaluate(
                        student,
                        teacher,
                        student_tokenizer,
                        teacher_tokenizer,
                        eval_dataset,
                        device,
                        args.max_length,
                        args.eval_batches,
                    )
                    if metrics["E_old"] is not None and metrics["E_zh"] is not None:
                        print(
                            f"[step {global_step}] E_old={metrics['E_old']:.4f} "
                            f"E_zh={metrics['E_zh']:.4f}"
                        )
                    else:
                        print(f"[step {global_step}] eval skipped (no samples)")

                if args.save_every and global_step % args.save_every == 0:
                    save_path = os.path.join(args.output_dir, f"step-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    student.save_pretrained(save_path)
                    student_tokenizer.save_pretrained(save_path)

                print(
                    f"[step {global_step}] loss={total_loss.item():.6f} "
                    f"zh={loss_zh.item():.6f} align={loss_align.item():.6f} "
                    f"retain={loss_retain.item():.6f}"
                )

        if args.max_steps and global_step >= args.max_steps:
            break

    student.save_pretrained(args.output_dir)
    student_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
