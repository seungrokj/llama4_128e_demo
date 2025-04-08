import argparse

import gradio as gr
from openai import OpenAI
import base64

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8000/v1',
                    help='Model URL')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    #required=True,
                    required=False,
                    help='Model name for the chatbot')
parser.add_argument('--temp',
                    type=float,
                    default=0.8,
                    help='Temperature for text generation')
parser.add_argument('--stop-token-ids',
                    type=str,
                    default='',
                    help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def predict(message, history):
    image_url = ""
    if len(message["files"]) != 0:
        #image_url = message["files"][0]["url"]
        image_url = message["files"][0]
    # Convert chat history to OpenAI format
    history_openai_format = [{
        "role": "user",
        "content": "You are a great ai assistant."
    }]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({
            "role": "assistant",
            "content": assistant
        })

    if image_url == "": 
        stream = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": message["text"]
            }],
            model=model,
            max_tokens=1024,
            stream=True
        )
    else:
        base64_image = encode_image(image_url)
        stream = client.chat.completions.create(
            messages=[{ 
                        "role": "user",
                         "content": [
                            { "type": "text", 
                              "text": message["text"]
                            },
                            { "type": "image_url", 
                              "image_url": {
                                  "url": f"data:image/jpeg;base64,{base64_image}"
                                  },
                            },
                        ]
                    }],
            model=model,
            max_tokens=1024,
            stream=True
        )

    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message


def echo(message, history):
    print(message["files"][0]["url"])
    return message["text"]

demo = gr.ChatInterface(
    fn=predict,
    title="Welcome to the Multi-modal Chatbot powered by AMD MI300X GPU and Llama-4-Maverick-17B-128E-Instruct 🌟",
    multimodal=True,
)
demo.queue().launch(server_name=args.host,
                     server_port=args.port,
                     share=True)
