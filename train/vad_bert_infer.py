from transformers import AutoModelForSequenceClassification, AutoTokenizer

ori_tokenizer = AutoTokenizer.from_pretrained("RobroKools/vad-bert")
ori_model = AutoModelForSequenceClassification.from_pretrained("RobroKools/vad-bert")

# tokenizer = AutoTokenizer.from_pretrained("train/lora_vad_bert_merged_step-150")
# model = AutoModelForSequenceClassification.from_pretrained("train/lora_vad_bert_merged_step-150")
tokenizer = AutoTokenizer.from_pretrained("train/xlm_roberta_emobank_next")
model = AutoModelForSequenceClassification.from_pretrained("train/xlm_roberta_emobank_next")

while True:
    text = input("Input: ").strip()
    if not text or text.lower() in {"quit", "exit"}:
        break
    
    inputs = ori_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = ori_model(**inputs)
    ori_vad = outputs.logits.detach().squeeze().tolist()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    vad = outputs.logits.detach().squeeze().tolist()

    print("ori_VAD:", ori_vad)
    print("xlm_VAD:", vad)
    print("   Diff:", [n - o for n, o in zip(vad, ori_vad)])
