import argparse
import base64
import json
import os
import random
from pathlib import Path

import huggingface_hub
import requests
import torch
from dotenv import dotenv_values
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    pipeline,
)

from src import BASE_DIR
from src.data_loader import BiGGenBenchLoader

DUMMY = False


def qwen_completions():
    torch.manual_seed(1234)

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-VL-Chat", trust_remote_code=True
    )

    # use bf16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
    # use cuda device
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True
    ).eval()

    # Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
    # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    # 1st dialogue turn
    query = tokenizer.from_list_format(
        [
            {
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
            {"text": "这是什么"},
        ]
    )
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
    # 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种可能是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，似乎在和人类击掌。两人之间充满了信任和爱。

    # 2nd dialogue turn
    response, history = model.chat(tokenizer, '输出"击掌"的检测框', history=history)
    print(response)
    # <ref>击掌</ref><box>(517,508),(589,611)</box>
    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if image:
        image.save("1.jpg")
    else:
        print("no box")


def save_checkpoint(outputs, filepath="checkpoint.json"):
    """Saves the outputs list to a file."""
    with open(filepath, "w") as f:
        json.dump(outputs, f)


def load_checkpoint(filepath="checkpoint.json"):
    """Loads the outputs list from a file if it exists."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        return []


def encode_image(image_path):
    path = Path(image_path)

    with path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def dummy_completions(inputs, **kwargs):
    return ["dummy output"] * len(inputs)


def test():
    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto")

    image1 = Image.open(
        requests.get(
            "https://llava-vl.github.io/static/images/view.jpg", stream=True
        ).raw
    )
    image2 = Image.open(
        requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
        ).raw
    )

    prompts = [
        "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
        "USER: <image>\nPlease describe this image\nASSISTANT:",
    ]

    inputs = processor(
        prompts, images=[image1, image2], padding=True, return_tensors="pt"
    ).to("cuda")
    for k, v in inputs.items():
        print(k, v.shape)

    output = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    for text in generated_text:
        print(text.split("ASSISTANT:")[-1])

    pipe = pipeline("image-to-text", model=model_id)

    prompt = "USER: <image>\nWhat are the things I should be cautious about when I visit this place?\nASSISTANT:"

    outputs = pipe(image1, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    print(outputs[0]["generated_text"])


def hf_vlm_completions(model_name, records, params, checkpoint_file="checkpoint.json"):
    if "1.5" in model_name:
        processor = AutoProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        )
    elif "1.6" in model_name:
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

    outputs = load_checkpoint(checkpoint_file)

    start_index = len(outputs)

    print(f"Running infernece of model {model_name} from index {start_index}")

    # Wrap the loop with tqdm for a progress bar
    for index in tqdm(
        range(start_index, len(records)),
        initial=start_index,
        total=len(records),
        desc="Processing Messages",
    ):
        record = records[index]
        max_attempts = 5  # Set a max attempts to avoid infinite loop

        try:
            prompts = [record["input_str"]]
            image = Image.open(str(record["image_path"]))
            # if "1.6" not in model_name:
            if True:
                inputs = processor(
                    prompts, images=[image], padding=True, return_tensors="pt"
                ).to("cuda")
                output = model.generate(**inputs, **params)
                generated_text = processor.batch_decode(
                    output, skip_special_tokens=True
                )
                response = generated_text[0].split("ASSISTANT:")[-1].strip()
            else:
                inputs = processor(
                    prompts[0], image=image, padding=True, return_tensors="pt"
                ).to("cuda")
                output = model.generate(**inputs, **params)
                generated_text = processor.decode(output[0], skip_special_tokens=True)
                response = generated_text.replace(record["input_str"], "").strip()
                import pdb

                pdb.set_trace()
            outputs.append(response)
            save_checkpoint(outputs, checkpoint_file)
        except Exception as e:
            print(f"Error processing message at index {index}: {e}")
            save_checkpoint(outputs, checkpoint_file)
            raise

    return outputs


def prepare_inputs(records, model_name: str):
    inputs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    for record in records:
        input_str = (
            f"USER: <image>\n{record['system_prompt']}\n{record['input']}\nASSISTANT:"
        )
        inputs.append(input_str)

    random_inputs = random.sample(inputs, 3)
    width = 20

    for input_str in random_inputs:
        print("-" * width)
        print("Example inputs:")
        print(input_str)
    print("-" * width)

    return inputs


def main(model_name: str, force_rerun: bool):
    loader = BiGGenBenchLoader()
    loader.load_data()
    dataset_dict = loader.data

    uid2data = {}

    for capability_name, tasks in dataset_dict.items():
        if capability_name not in ["vision"]:
            continue
        for task_name, instances in tasks.items():
            for instance in instances:
                uid2data[instance["id"]] = instance

    image_base_path = loader.dataset_path / "vision"
    for uid, instance in uid2data.items():
        image_path = (
            image_base_path / instance["task"] / f"{instance['instance_idx']}.jpg"
        )
        uid2data[uid]["image_path"] = image_path

    uid2data = dict(sorted(uid2data.items(), key=lambda item: item[0]))
    data_list = list(uid2data.values())

    inputs = prepare_inputs(data_list, model_name)

    for input_str, data in zip(inputs, data_list):
        data["input_str"] = input_str

    params = {
        "max_new_tokens": 2048,
        "do_sample": True,
        # "n": 1,
        "temperature": 1.0,
        "top_p": 0.9,
    }

    data_path = os.path.join(os.path.dirname(__file__), "responses_vision")
    output_path = Path(data_path) / f"{model_name.split('/')[-1]}_responses.json"

    if output_path.exists() and not force_rerun:
        print("Output file already exists. Run Finished.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_checkpoint = os.path.join(
        BASE_DIR,
        f"cache_responses/vlm/{model_name.split('/')[-1]}_checkpoint.json",
    )

    Path(model_checkpoint).parent.mkdir(parents=True, exist_ok=True)

    if DUMMY:
        outputs = dummy_completions(data_list, **params)
    else:
        outputs = hf_vlm_completions(
            model_name, data_list, params, checkpoint_file=model_checkpoint
        )

    results = {}

    for idx, data in enumerate(data_list):
        uid2data[data["id"]].update({"response": outputs[idx].strip()})
        results[data["id"]] = {
            "capability": data["capability"],
            "task": data["task"],
            "response": data["response"].strip(),
        }

    with output_path.open("w") as file:
        file.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Name of the model to evaluate",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Use system prompt during evaluation",
    )

    args = parser.parse_args()

    hf_token = dotenv_values(".env")["HF_TOKEN"]
    huggingface_hub.login(token=hf_token)

    main(args.model_name, True)
