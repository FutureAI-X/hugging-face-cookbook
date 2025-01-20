from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

classifier = pipeline("sentiment-analysis", device=0, batch_size=1)
result = classifier(
    [
        "I absolutely love the new design of this app!",
        "I absolutely love the new design of this app!",
        "I absolutely love the new design of this app!",
        "I absolutely love the new design of this app!",
        "I absolutely love the new design of this app!",
        "I absolutely love the new design of this app!",
        "I absolutely love the new design of this app!",
        "I absolutely love the new design of this app!",
        "I absolutely love the new design of this app!",
        "I absolutely love the new design of this app!"
    ]
)

print(result)