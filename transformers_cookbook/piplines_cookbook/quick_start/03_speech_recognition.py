from transformers import pipeline
from datasets import load_dataset, Audio

speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

dataset = load_dataset("./minds14", name="en-US", split="train")

# 音频文件在调用 "audio" 列时自动加载和重采样
dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

result = speech_recognizer(dataset[:10]["audio"])
print([d for d in result])