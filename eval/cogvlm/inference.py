import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, json
from tqdm import tqdm
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from dataset.dataloader import MathLoader


MODEL_PATH = "./checkpoints/cogvlm2"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True,
).to(DEVICE).eval()

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

dataloader = MathLoader("./data/pazhou", "./data/pazhou/test.jsonl")
# set the max number of tiles in `max_num`
gen_kwargs = {
    "max_new_tokens": 2048,
    "pad_token_id": 128002,
}
with open("./results/cogvlm2_exact.txt", "w") as f:
    for data in tqdm(dataloader):
        img_path, id, question = data['image_path'], data['question_id'], data['question']

        image = Image.open(img_path).convert('RGB')

        history = []

        query = "USER: {} ASSISTANT:".format(question)
        print("-" * 10, "\nquery 1: ", query)

        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            images=[image],
            template_version='chat'
        )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
        }


        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]
            print("\nCogVLM2:", response)
        history.append((query, response))

        if "options" in question:
            options = question.split("options: ")[1]
            question = f'Extract the direct answer from model response with a single option, options from the following options: {options}" Answer:'
        else:
            question = 'Extract the direct answer from model response, which is a single number value, formula or short sentence. Answer:'

        query = "USER: {} ASSISTANT:".format(question)
        print("-"*10, "\nquery 2: ", query)

        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            images=[image],
            template_version='chat'
        )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]
            print("\nCogVLM2:", response)
        f.write(json.dumps({"id": id, "model_answer": response}) + "\n")
