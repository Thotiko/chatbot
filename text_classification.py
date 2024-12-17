from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import warnings

warnings.filterwarnings("ignore")

# Path to the trained Vietnamese sentiment model
model_path = "./vietnamese_sentiment_model/vietnamese_sentiment_model"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def get_sentiment_label(query):
    """
    Thực hiện phân tích cảm xúc và trả về nhãn cảm xúc.

    Parameters:
        query (str): Câu văn cần phân tích cảm xúc.

    Returns:
        str: Nhãn cảm xúc tương ứng ('positive', 'negative', hoặc 'neutral').
    """
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Map class to lowercase sentiment labels
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map[predicted_class]

# Test examples
# if __name__ == "__main__":
#     examples = [
#         "Giảng viên dạy rất hay và tận tình",
#         "Bài giảng khó hiểu quá",
#         "Lớp học bình thường"
#     ]
#     for text in examples:
#         sentiment = get_sentiment_label(text)
#         print(f"Text: {text}\nSentiment: {sentiment}\n")