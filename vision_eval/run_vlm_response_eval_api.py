import asyncio
import base64
import glob
import json
import os
import random
import time
import warnings
from pathlib import Path

import huggingface_hub
from dotenv import dotenv_values
from litellm import completion
from tqdm import tqdm

from src.data_loader import BiGGenBenchLoader
from src.llms.openai_utils import OpenAILLM
from src.prompts import ABS_SYSTEM_PROMPT, SCORE_RUBRIC_TEMPLATE

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

DEBUG = False
DUMMY = False


async def dummy_async_completions(inputs, **kwargs):
    ["Hello. [RESULT] 5"] * len(inputs)
    await asyncio.sleep(1)
    return ["Hello. [RESULT] 5"] * len(inputs)


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


def dummy_completions(inputs, **kwargs):
    return ["dummy output"] * len(inputs)


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


async def batch_completions_with_retries(
    model,
    inputs,
    params,
    batch_size,
    parse_output,
    max_retries=5,
):
    # DEBUG: Debugging purposes
    if DEBUG:
        inputs = inputs[:10]
    batched_outputs = []

    total_batches = len(inputs) // batch_size + (
        1 if len(inputs) % batch_size > 0 else 0
    )
    total_len = len(inputs)

    # Process initial batches with progress bar
    print("Processing initial batches...")
    for i in tqdm(
        range(0, len(inputs), batch_size), total=total_batches, desc="Initial Batches"
    ):
        batch_inputs = inputs[i : i + batch_size]
        if DUMMY:
            batch_outputs = await dummy_async_completions(batch_inputs, **params)
        else:
            batch_outputs = await model.completions(batch_inputs, **params)
        batched_outputs.extend(batch_outputs)

    # Identify failed instances and prepare for retries
    to_retry_inputs = []
    to_retry_indices = []
    for i, output in enumerate(batched_outputs):
        feedback, score = parse_output(output)
        if feedback is None:  # Parsing failed
            # DEBUG: Debugging purposes
            if DEBUG:
                print("Parsing failed: ", output)
                import pdb

                pdb.set_trace()
            to_retry_inputs.append(inputs[i])
            to_retry_indices.append(i)

    # Retry logic with progress bar
    retries = 0
    while to_retry_inputs and retries < max_retries:
        retries += 1
        print(f"Retrying failed batches: Attempt {retries}/{max_retries}")
        retry_outputs = []
        for i in tqdm(
            range(0, len(to_retry_inputs), batch_size), desc=f"Retry Attempt {retries}"
        ):
            batch_inputs = to_retry_inputs[i : i + batch_size]
            if DUMMY:
                batch_outputs = await dummy_async_completions(
                    batch_inputs, **params, use_tqdm=True
                )
            else:
                batch_outputs = await model.completions(batch_inputs, **params)

            assert len(batch_outputs) == len(batch_inputs)
            retry_outputs.extend(batch_outputs)

        new_to_retry_inputs = []
        new_to_retry_indices = []
        for idx, (retry_idx, output) in enumerate(zip(to_retry_indices, retry_outputs)):
            feedback, score = parse_output(output)
            if feedback is None:  # Still failing
                new_to_retry_inputs.append(to_retry_inputs[idx])
                new_to_retry_indices.append(to_retry_indices[idx])
            else:
                batched_outputs[retry_idx] = output  # Update with successful retry

        to_retry_inputs = new_to_retry_inputs
        to_retry_indices = new_to_retry_indices

    # Final aggregation and printing
    outputs_len = len(batched_outputs)
    print(f"Processed {outputs_len}/{total_len} instances.")

    if outputs_len < total_len:
        warnings.warn("Some instances failed to generate feedback.")
        warnings.warn("They will be written as None in the output file.")
        raise Exception(
            f"Failed to generate feedback for {total_len - outputs_len} instances."
        )

    feedbacks = []
    scores = []

    for output in tqdm(batched_outputs, desc="Finalizing"):
        feedback, score = parse_output(output)
        if feedback is not None:
            feedbacks.append(feedback)
            scores.append(score)
        else:
            # raise Exception(
            #     f"Parsing failed for output: {output}. Feedback: {feedback}, Score: {score}"
            # )
            feedbacks.append(None)
            scores.append(None)

    if DEBUG:
        print("Checking the results")
        print(*list(zip(feedbacks, scores))[:10])
        # print(*feedbacks[:10])
        # print(*scores[:10])
        # pdb.set_trace()

    return feedbacks, scores


def prepare_inputs(records, model_name: str):
    assert model_name == "gpt-4-vision-preview", "Model not supported"

    inputs = []

    for record in records:
        orig_response = record["response"]
        orig_instruction = record["input"]
        score_rubric = SCORE_RUBRIC_TEMPLATE.format(**record["score_rubric"]).strip()
        system_message = ABS_SYSTEM_PROMPT
        orig_reference_answer = record["reference_answer"]

        content = ABS_VLM_PROMPT.format(
            orig_response=orig_response,
            orig_instruction=orig_instruction,
            orig_reference_answer=orig_reference_answer,
            score_rubric=score_rubric,
        )

        base64_image = encode_image(record["image_path"])
        image_url = f"data:image/jpeg;base64,{base64_image}"

        message = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
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


async def main(model_name: str, force_rerun: bool):
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
        uid2data[uid]["image_path"] = str(image_path)

    uid2data = dict(sorted(uid2data.items(), key=lambda item: item[0]))
    # data_list = list(uid2data.values())

    response_path = os.path.join(os.path.dirname(__file__), "responses_vision")
    pattern = os.path.join(response_path, "*.json")
    json_files = glob.glob(pattern)

    for file_path in json_files:
        print(f"Loading file: {file_path}")
        response_model_name = file_path.split("/")[-1].replace("_responses.json", "")
        response_dict = uid2data.copy()

        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        for id, record in data.items():
            if id not in uid2data.keys():
                if "llm_judge" not in id:
                    continue
            response = record["response"]
            response = response.split("[/INST]")[-1].strip()
            response_dict[id].update({"response": response})

        response_dict = dict(sorted(response_dict.items(), key=lambda item: item[0]))
        data_list = list(uid2data.values())
        model = OpenAILLM(model_name)
        # model = None
        inputs = prepare_inputs(data_list, model_name)

        params = {
            "max_tokens": 2048,
            "n": 1,
            "temperature": 1.0,
            "top_p": 0.9,
        }

        data_path = os.path.join(os.path.dirname(__file__), "responses_gpt4v_eval")
        output_path = Path(data_path) / f"{response_model_name}_evaluation.json"

        if output_path.exists() and not force_rerun:
            print("Output file already exists. Run Finished.")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        batch_size = 10

        feedbacks, scores = await batch_completions_with_retries(
            model, inputs, params, batch_size, parse_output
        )

        assert len(feedbacks) == len(scores)
        assert len(feedbacks) == len(data_list)

        for idx, instance in enumerate(data_list):
            response_dict[instance["id"]].update(
                {"feedback": feedbacks[idx], "score": scores[idx]}
            )

        with output_path.open("w") as file:
            file.write(json.dumps(response_dict, indent=4))


if __name__ == "__main__":
    model_name = "gpt-4-vision-preview"

    hf_token = dotenv_values(".env")["HF_TOKEN"]
    huggingface_hub.login(token=hf_token)

    asyncio.run(main(model_name, False))
