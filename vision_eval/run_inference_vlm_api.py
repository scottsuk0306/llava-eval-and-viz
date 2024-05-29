import argparse
import base64
import json
import os
import random
import time
from pathlib import Path

import huggingface_hub
import requests
from anthropic import Anthropic
from dotenv import dotenv_values, load_dotenv
from litellm import completion
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from src.data_loader import BiGGenBenchLoader

DUMMY = False


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

    AutoProcessor.from_pretrained(model_id)
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


def litellm_completions(model_name, inputs, params, checkpoint_file="checkpoint.json"):
    outputs = load_checkpoint(checkpoint_file)

    start_index = len(outputs)

    print(f"Running infernece of model {model_name} from index {start_index}")

    # Wrap the loop with tqdm for a progress bar
    for index in tqdm(
        range(start_index, len(inputs)),
        initial=start_index,
        total=len(inputs),
        desc="Processing Messages",
    ):
        message = inputs[index]
        attempt = 0
        max_attempts = 5  # Set a max attempts to avoid infinite loop

        while True:
            try:
                response = completion(model=model_name, messages=message, **params)
                outputs.append(response["choices"][0]["message"]["content"])
                save_checkpoint(outputs, checkpoint_file)
                break
            except Exception as e:
                print(
                    f"Error processing message at index {index} on attempt {attempt + 1}: {e}"
                )
                # import pdb; pdb.set_trace()
                attempt += 1
                if attempt >= max_attempts:
                    print(
                        f"Failed to process message at index {index} after {max_attempts} attempts."
                    )
                    outputs.append(
                        "[ERROR] Failed to process message after max attempts."
                    )
                    save_checkpoint(outputs, checkpoint_file)
                    break  # Exit the loop after reaching max attempts

                # Optional: Implement a backoff strategy before retrying
                time.sleep(1)

    return outputs


def anthropic_completions(
    model_name, inputs, params, checkpoint_file="checkpoint.json"
):
    client = Anthropic()
    outputs = load_checkpoint(checkpoint_file)

    start_index = len(outputs)

    print(f"Running inferenece of model {model_name} from index {start_index}")

    # Wrap the loop with tqdm for a progress bar
    for index in tqdm(
        range(start_index, len(inputs)),
        initial=start_index,
        total=len(inputs),
        desc="Processing Messages",
    ):
        message = inputs[index]
        max_attempts = 5  # Set a max attempts to avoid infinite loop

        try:
            response = client.messages.create(
                model=model_name,
                system=message[0]["content"],
                messages=message[1:],
                **params,
            )
            outputs.append(response["choices"][0]["message"]["content"])
            save_checkpoint(outputs, checkpoint_file)
        except Exception as e:
            print(f"Error processing message at index {index}: {e}")
            save_checkpoint(outputs, checkpoint_file)
            raise

    return outputs


def prepare_inputs(records, model_name: str):
    inputs = []

    for record in records:
        base64_image = encode_image(record["image_path"])
        if model_name in ["gemini/gemini-pro-vision"]:
            image_url = str(record["image_path"])
        else:
            image_url = f"data:image/jpeg;base64,{base64_image}"

        message = [
            {"role": "system", "content": record["system_prompt"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": record["input"]},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        inputs.append(message)

    random_inputs = random.sample(inputs, 3)
    width = 20

    for input_str in random_inputs:
        print("-" * width)
        print("Example inputs:")
        print(input_str[1]["content"][0]["text"])
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

    # import pdb; pdb.set_trace()

    inputs = prepare_inputs(data_list, model_name)

    params = {
        "max_tokens": 2048,
        "n": 1,
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
        os.path.dirname(__file__),
        f"../cache_responses/vlm/{model_name.split('/')[-1]}_checkpoint.json",
    )

    Path(model_checkpoint).parent.mkdir(parents=True, exist_ok=True)

    if DUMMY:
        outputs = dummy_completions(inputs, **params)
    else:
        outputs = litellm_completions(
            model_name, inputs, params, checkpoint_file=model_checkpoint
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
        default="gemini/gemini-pro-vision",
        help="Name of the model to evaluate",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Use system prompt during evaluation",
    )

    args = parser.parse_args()

    load_dotenv()

    # litellm.set_verbose = True

    hf_token = dotenv_values(".env")["HF_TOKEN"]
    huggingface_hub.login(token=hf_token)

    main(args.model_name, True)
