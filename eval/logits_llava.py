from PIL import Image
import requests, pdb
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("./checkpoints/llava-v1.5-13b")
processor = AutoProcessor.from_pretrained("./checkpoints/llava-v1.5-13b")

prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=15, return_dict=True)
print(type(generate_ids))
print(generate_ids.logits)
pdb.set_trace()

processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]