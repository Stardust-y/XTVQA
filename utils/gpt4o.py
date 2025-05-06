import base64
# from openai import OpenAI
import openai
openai.api_base = "https://openkey.cloud/v1"
openai.api_key = "sk-EQeEiftqrY7dPEqQ7eCcBaAb48B64d83B8F23045D9652153"
IMAGE_PATH = "./data/docvqa/documents/ffgf0001_2.png"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

MODEL="gpt-4o"
base64_image = encode_image(IMAGE_PATH)


response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
    ],
    temperature=0.0,
    max_tokens=512,
    n=5
    )
print(response.choices[0].message.content)
