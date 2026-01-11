import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load trained model
tokenizer = AutoTokenizer.from_pretrained("news_classifier")
model = AutoModelForSequenceClassification.from_pretrained("news_classifier")
model.eval()

labels = ["World", "Sports", "Business", "Sci/Tech"]

def categorize_news(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    return {
        "category": labels[predicted_class.item()],
        "confidence": round(confidence.item(), 3)
    }

# Example
if __name__ == "__main__":
    article = """
    Apple announced a major investment in artificial intelligence,
    unveiling new chips designed for high-performance computing.
    """

    print(categorize_news(article))
