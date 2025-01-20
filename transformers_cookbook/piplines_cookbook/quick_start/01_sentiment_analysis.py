from transformers import pipeline

classifier = pipeline('sentiment-analysis')

results_single  = classifier('I absolutely love the new design of this app!')
print(f"单个模式：{results_single}")

results_batch  = classifier(['I absolutely love the new design of this app!', 'This app is terrible, I hate it.'])
print(f"批量模式：{results_batch}")

