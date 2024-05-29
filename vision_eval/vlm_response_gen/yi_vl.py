import json
import os

import torch
from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from PIL import Image
from tqdm import tqdm


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


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

disable_torch_init()
model_path = "01-ai/Yi-VL-34B"
key_info["model_path"] = model_path
get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path)

output_list = []
for d in tqdm(data_list):
    image_file = f"../tasks/vision/{d['task']}/{str(d['instance_idx'])}.jpg"
    qs = d["system_prompt"] + "\n\n" + d["input"]
    # image_file = args.image_file
    # qs = args.question
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates["mm_default"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    image = Image.open(image_file)
    if getattr(model.config, "image_aspect_ratio", None) == "pad":
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean)
        )
    image_tensor = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ][0]

    stop_str = conv.sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    model = model.to(dtype=torch.bfloat16)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            stopping_criteria=[stopping_criteria],
            max_new_tokens=2048,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    d["yi_vl_34b_response"] = outputs
    output_list.append(d)


with open(f"./results/yi_vl_34b.json", "w") as f:
    json.dump(output_list, f, indent=4)
