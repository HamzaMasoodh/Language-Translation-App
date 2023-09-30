import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained(model_name)

translation_pipeline = pipeline('translation_en_to_de', model=model, tokenizer=tokenizer)

def translate_transformers(from_text):
    results = translation_pipeline(from_text)
    return results[0]['translation_text']

interface = gr.Interface(
    fn=translate_transformers, 
    inputs=gr.Textbox(lines=2, placeholder='Text to translate'),
    outputs='text',
    title="Language Translator"
)

interface.launch()
