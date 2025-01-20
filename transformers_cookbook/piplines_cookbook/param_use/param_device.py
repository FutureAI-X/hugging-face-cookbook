from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis", device=0)
result = classifier(["这家餐厅的菜味道非常棒！", "I absolutely love the new design of this app!"])

print(result)