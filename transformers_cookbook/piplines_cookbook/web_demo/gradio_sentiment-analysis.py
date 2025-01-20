from transformers import pipeline
import gradio as gr

classifier = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis", device=0)

gr.Interface.from_pipeline(classifier).launch(share=True)