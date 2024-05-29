import base64
import glob
import json
import os
import random
import time
from pathlib import Path

import huggingface_hub
from dotenv import dotenv_values

from src.data_loader import BiGGenBenchLoader
from src.prompts import SCORE_RUBRIC_TEMPLATE

DEBUG = False
DUMMY = False


ABS_VLM_PROMPT = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, image and a score rubric representing an evaluation criterion is given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Reference Answer (Score 5):
{orig_reference_answer}

###Score Rubrics:
{score_rubric}

###Feedback: """


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


def dummy_completions(inputs, **kwargs):
    ["Hello. [RESULT] 5"] * len(inputs)
    time.sleep(1)
    return ["Hello. [RESULT] 5"] * len(inputs)


def parse_output(outputs):
    parts = outputs.split("[RESULT]")
    if len(parts) == 2:
        feedback, result = parts[0].strip(), parts[1].strip()
        if result.isdigit() and result in ["1", "2", "3", "4", "5"]:
            return feedback, int(result)
    return None, None


def encode_image(image_path):
    path = Path(image_path)

    with path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def prepare_inputs(records, model_name: str):
    assert model_name == "kaist-ai/prometheus-vision-13b-v1.0", "Model not supported"

    inputs = []

    for record in records:
        orig_response = record["response"]
        orig_instruction = record["input"]
        score_rubric = SCORE_RUBRIC_TEMPLATE.format(**record["score_rubric"]).strip()
        orig_reference_answer = record["reference_answer"]

        content = ABS_VLM_PROMPT.format(
            orig_response=orig_response,
            orig_instruction=orig_instruction,
            orig_reference_answer=orig_reference_answer,
            score_rubric=score_rubric,
        )

        input_str = content
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
    assert model_name in [
        "kaist-ai/prometheus-vision-13b-v1.0",
    ], "Model not supported"

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
        uid2data[uid]["image"] = str(image_path)

    uid2data = dict(sorted(uid2data.items(), key=lambda item: item[0]))
    data_list = list(uid2data.values())

    response_path = os.path.join(os.path.dirname(__file__), "responses_vision")
    pattern = os.path.join(response_path, "*.json")
    json_files = glob.glob(pattern)

    for file_path in json_files:
        eval_data_list = []

        print(f"Loading file: {file_path}")
        response_model_name = file_path.split("/")[-1].replace("_responses.json", "")

        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        assert len(data) == 50

        for id, record in data.items():
            if id not in uid2data.keys():
                assert False, f"ID {id} not found in the dataset"
            uid2data[id].update({"response": record["response"]})

        inputs = prepare_inputs(data_list, model_name)

        for input_str, data in zip(inputs, data_list):
            data["instruction"] = input_str

        data_path = os.path.join(
            os.path.dirname(__file__), "responses_prometheus_vision_eval"
        )
        output_path = (
            Path(data_path) / "eval_data" / f"{response_model_name}_eval_data.json"
        )

        if output_path.exists() and not force_rerun:
            print("Output file already exists. Run Finished.")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data_list = list(uid2data.values())

        for instance in data_list:
            eval_data_list.append(
                {
                    "image": instance["image"],
                    "text": instance["instruction"].strip(),
                    "question_id": instance["id"],
                }
            )

        assert len(eval_data_list) == 50

        with output_path.open("w") as file:
            for i, record in enumerate(eval_data_list):
                file.write(json.dumps(record) + "\n")

        # MULTI_RUN = False

        # if MULTI_RUN:
        #     all_scores = [scores]
        #     for num_trial in range(1, 5):
        #         print(f"Trial {num_trial}")
        #         _, temp_scores = batch_completions_with_retries(
        #             model, inputs, params, batch_size, parse_output
        #         )
        #         all_scores.append(temp_scores)
        #     zipped_scores = list(zip(*all_scores))
        #     combined_scores = [list(score_group) for score_group in zipped_scores]
        #     assert len(combined_scores) == len(scores)
        #     scores = combined_scores


if __name__ == "__main__":
    model_name = "kaist-ai/prometheus-vision-13b-v1.0"

    hf_token = dotenv_values(".env")["HF_TOKEN"]
    huggingface_hub.login(token=hf_token)

    main(model_name, True)
