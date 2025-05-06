import os
import pdb
import time

import google.generativeai as genai
from pathlib import Path

generation_config = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safe = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]


# verify response
def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        # print("Response Error")
        return False
    return True


class Gemini_Model():
    def __init__(self, key, vision=True, patience=1, sleep_time=1):
        print("!!Bard Token:", key)
        self.patience = patience
        self.sleep_time = sleep_time
        genai.configure(api_key=key)
        self.vision = vision
        if self.vision:
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                               generation_config=generation_config,
                                               safety_settings=safe)
        else:
            self.model = genai.GenerativeModel(model_name="gemini-pro",
                                               generation_config=generation_config,
                                               safety_settings=safe)

    def get_response_text(self, prompt):
        response = self.model.generate_content(prompt)
        return response

    def get_response_vision(self, image_path, input_text):
        patience = self.patience
        while patience > 0:
            patience -= 1
            # print(f"Patience: {patience}")
            try:
                # Validate that an image is present
                print("img", image_path)
                if not (img := Path(image_path)).exists():
                    raise FileNotFoundError(f"Could not find image: {img}")

                print(type(Path(image_path).read_bytes()))

                image_parts = [
                    {
                        "mime_type": "image/jpeg",
                        "data": Path(image_path).read_bytes()
                    },
                ]

                prompt_parts = [
                    input_text,
                    image_parts[0],
                ]

                response = self.model.generate_content(prompt_parts)

                return response.text


            except Exception as e:
                print(e)
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
        return ""


if __name__ == "__main__":
    model = Gemini_Model(key="AIzaSyCYo6MWJKX4nrV8i36GKVVEVeuYfD3co-s", vision=False)
    response = model.get_response_text("hello")
    print(response)