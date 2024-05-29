import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

torch.manual_seed(1234)
import json
import os

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True, cache_dir="./cache"
).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(
    "Qwen/Qwen-VL-Chat", trust_remote_code=True
)

root_dir = "../tasks/vision"
data_list = []
# Walk through all subdirectories of the root directory
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # Check if the current file is 'data.json'
        if file == "data.json":
            # Construct the full file path
            file_path = os.path.join(subdir, file)
            # Open and load the json file
            with open(file_path, "r") as json_file:
                data = json.load(json_file)

                for d in data:
                    data_list.append(d)

output_list = []
for d in data_list:
    query = tokenizer.from_list_format(
        [
            {
                "image": f"../tasks/vision/{d['task']}/{str(d['instance_idx'])}.jpg"
            },  # Either a local path or an url
            {"text": d["system_prompt"] + "\n\n" + d["input"]},
        ]
    )
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
    d["qwen_vl_response"] = response
    output_list.append(d)

with open(f"./results/qwen_vl.json", "w") as f:
    json.dump(output_list, f, indent=4)
