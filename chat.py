import gradio as gr
from llama_cpp import Llama

llm = Llama(
    model_path="models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
    chat_format="llama-3",
    n_ctx=2048,
)


def predict(message, history):
    messages = [{"role": "user", "content": message}]
    for human_content, system_content in history:
        messages.append({"role": "user", "content": human_content})
        messages.append({"role": "system", "content": system_content})

    response = llm.create_chat_completion(messages)

    return response['choices'][0]['message']['content']



llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。",
        },
    ],
    max_tokens=512,
)


iface = gr.ChatInterface(fn=predict)
iface.launch()
