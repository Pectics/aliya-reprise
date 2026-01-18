from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("RobroKools/vad-bert")
model = AutoModelForSequenceClassification.from_pretrained("RobroKools/vad-bert")

while True:
    text = input("Enter text: ")
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    vad = outputs.logits.detach().squeeze().tolist()
    print("VAD: ", vad)  # [valence, arousal, dominance]
